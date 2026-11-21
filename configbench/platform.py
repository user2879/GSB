"""平台编排器：节点生命周期、增量编译/下发、双路径一致性、延迟预算与恢复。"""

import json
import threading
import time
from dataclasses import dataclass, field

from . import fixedpoint
from .aggregate import aggregate_slice, slice_digest
from .compiler import build_node_slices, compile_version
from .errors import (FieldError, LatencyError, LedgerError, PortError,
                     PreviewError, ReplayError)
from .ledger import LEDGER_FORMAT_VERSION, Ledger
from .manifest import ItemSpec
from .scheduler import Scheduler, Task, mono_ms
from .transport import NodeProcess, dispatch_once

# 权威重算耗时与账本观测耗时的允许误差界（整数毫秒）。
DEFAULT_TIMING_TOLERANCE_MS = 0


def _item_to_wire(item):
    return {
        "key": item.key,
        "value_type": item.value_type,
        "value": item.value,
        "scale": item.scale,
        "targets": list(item.targets),
        "derives_from": [
            {"key": e.key, "coef_raw": e.coef_raw, "coef_scale": e.coef_scale}
            for e in item.derives_from
        ],
    }


@dataclass
class NodeView:
    """界面增量缓存中的单节点视图（快速路径）。"""
    node_id: str
    config_version: int = 0
    summary_digest: str = ""
    summary: object = None
    progress: str = "IDLE"          # IDLE/DISPATCHING/CURRENT/FAILED
    last_task_id: str = ""
    last_elapsed_ms: int = 0
    last_outcome: str = ""


class Platform:
    def __init__(self, registry, manifest, sample_decl, ledger_path,
                 max_global=4, per_node_quota=2):
        self.registry = registry
        self.manifest = manifest
        self.sample_decl = sample_decl
        self.ledger = Ledger(ledger_path)
        self.max_global = max_global
        self.per_node_quota = per_node_quota
        self.node_processes = {}
        # 界面增量缓存：内存态，可由账本重建。
        self.cache = {nid: NodeView(nid) for nid in sorted(registry.nodes)}
        self.cache_lock = threading.Lock()
        self._req_seq = 0
        self._req_lock = threading.Lock()
        # 测试/实验用：在每次界面查询计时区内注入额外耗时（默认无）。
        self.pre_query_hook = None

    # ---------- 节点子进程编排 ----------
    def start_nodes(self, only=None):
        """按注册表启动节点；端口/握手类失败分别抛出 E_PORT/E_HANDSHAKE。"""
        failures = []
        for node_id in sorted(self.registry.nodes):
            if only and node_id not in only:
                continue
            node = self.registry.nodes[node_id]
            proc = NodeProcess(node)
            try:
                proc.start()
            except PortError:
                failures.append(node_id)
                with self.cache_lock:
                    self.cache[node_id].progress = "FAILED"
                raise
            self.node_processes[node_id] = proc
        return sorted(self.node_processes)

    def stop_nodes(self):
        for node_id in sorted(self.node_processes):
            self.node_processes[node_id].stop()
        self.node_processes.clear()

    def kill_node_hard(self, node_id):
        """强制终止指定节点（崩溃实验用）。"""
        self.node_processes[node_id].kill_hard()
        del self.node_processes[node_id]

    def _next_request_id(self, prefix):
        with self._req_lock:
            self._req_seq += 1
            return f"{prefix}-{self._req_seq:06d}"

    def _sample_wire(self):
        return {
            "seed": self.sample_decl.seed,
            "n_samples": self.sample_decl.n_samples,
            "n_groups": self.sample_decl.n_groups,
            "value_range": self.sample_decl.value_range,
            "declared_scale": self.sample_decl.declared_scale,
        }

    # ---------- 版本发布 ----------
    def publish_version(self, version_number):
        cur = self.manifest.version(version_number)
        prev = (self.manifest.version(version_number - 1)
                if any(v.version == version_number - 1 for v in self.manifest.versions)
                else None)
        artifacts, propagation = compile_version(prev, cur)
        # 清单快照与编译事件先完整提交（崩溃恢复据此判定编译是否完成）。
        self.ledger.append("manifest_snapshot", {
            "version": version_number,
            "budget_ms": self.manifest.latency_budget_ms,
            "items": {k: _item_to_wire(it) for k, it in sorted(cur.items.items())},
        })
        self.ledger.append("compile", {
            "version": version_number,
            "artifacts": {
                k: {"artifact_id": a.artifact_id, "content": a.content,
                    "targets": list(a.targets)}
                for k, a in sorted(artifacts.items())},
            "propagation": propagation,
        })
        slices = build_node_slices(cur, self.registry)
        # 增量下发：首版本全量；后续版本仅调度受影响节点。
        affected_nodes = (set(slices) if prev is None
                          else set(propagation["affected_nodes"]))
        tasks, task_specs = self._build_dispatch(
            version_number, slices, only_nodes=affected_nodes)
        with self.cache_lock:
            for task_id, spec in task_specs.items():
                view = self.cache[spec["node_id"]]
                view.progress = "DISPATCHING"
                view.last_task_id = task_id
        scheduler = Scheduler(self.max_global, self.per_node_quota)
        results = scheduler.run(tasks)
        receipts = self._record_results(version_number, results, task_specs)
        self.ledger.set_cursor(f"v{version_number}:cursor",
                               {"published_at_ms": mono_ms(),
                                "task_ids": sorted(task_specs)})
        return {"version": version_number, "propagation": propagation,
                "artifacts": artifacts, "results": results, "receipts": receipts,
                "peaks": scheduler.peaks}

    def _build_dispatch(self, version_number, slices, only_nodes):
        tasks = []
        task_specs = {}
        for node_id in sorted(slices):
            if node_id not in only_nodes:
                continue
            node = self.registry.nodes[node_id]
            items = slices[node_id]
            request_id = self._next_request_id(f"v{version_number}-{node_id}")
            task_id = f"v{version_number}:{node_id}"
            request = {
                "type": "slice",
                "request_id": request_id,
                "config_version": version_number,
                "items": [_item_to_wire(i) for i in sorted(items, key=lambda x: x.key)],
                "samples": self._sample_wire(),
            }
            task_specs[task_id] = {"node_id": node_id, "request": request,
                                   "items": items, "digest": slice_digest(version_number, items)}
            policy = self._policy_for(items)
            tasks.append(Task(task_id, node_id,
                              lambda n=node, r=request, to=policy.timeout_ms:
                              dispatch_once(n, r, ack_timeout_ms=to),
                              retries=policy.retries))
        return tasks, task_specs

    @staticmethod
    def _policy_for(items):
        # 同一切片内策略取最保守：最大超时与最大重试次数。
        timeout = max(i.dispatch.timeout_ms for i in items)
        retries = max(i.dispatch.retries for i in items)
        class _P:
            pass
        p = _P()
        p.timeout_ms = timeout
        p.retries = retries
        return p

    def _record_results(self, version_number, results, task_specs):
        receipts = {}
        success_count = 0
        for result in results:
            spec = task_specs[result.task_id]
            # 逐任务结局（含每次尝试）完整提交，失败节点不计入成功回执。
            self.ledger.append("task_outcome", {
                "task_id": result.task_id,
                "node_id": result.node_id,
                "version": version_number,
                "outcome": result.outcome,
                "attempts": result.attempts,
                "wait_started_ms": result.wait_started_ms,
                "wait_ended_ms": result.wait_ended_ms,
                "wait_queued_ms": result.wait_queued_ms,
                "total_elapsed_ms": result.total_elapsed_ms,
                "slice_digest": spec["digest"],
            })
            view = self.cache[spec["node_id"]]
            view.last_outcome = result.outcome
            if result.outcome == "SUCCESS" and result.receipt is not None:
                ack = result.receipt
                receipt = {
                    "task_id": result.task_id,
                    "node_id": result.node_id,
                    "version": version_number,
                    "request_id": ack["request_id"],
                    "config_version": ack["config_version"],
                    "summary": ack["summary"],
                    "summary_digest": ack["summary_digest"],
                    "elapsed_ms": ack["elapsed_ms"],
                }
                self.ledger.append("receipt", receipt)
                receipts[result.task_id] = receipt
                success_count += 1
                view.config_version = version_number
                view.summary = ack["summary"]
                view.summary_digest = ack["summary_digest"]
                view.last_elapsed_ms = ack["elapsed_ms"]
                view.progress = "CURRENT"
            else:
                view.progress = "FAILED"
        return {"success": success_count, "total": len(results),
                "success_task_ids": sorted(receipts)}

    # ---------- 界面增量缓存快速路径 ----------
    def query_preview(self, request_id=None):
        """一次界面查询：只读增量缓存，记录单调时钟延迟并判定预算。"""
        request_id = request_id or self._next_request_id("q")
        start = mono_ms()
        if self.pre_query_hook is not None:
            self.pre_query_hook()
        with self.cache_lock:
            snapshot = [
                {"node_id": v.node_id, "config_version": v.config_version,
                 "summary_digest": v.summary_digest, "progress": v.progress,
                 "last_outcome": v.last_outcome,
                 "last_elapsed_ms": v.last_elapsed_ms,
                 "summary": v.summary}
                for v in (self.cache[nid] for nid in sorted(self.cache))
            ]
        elapsed = mono_ms() - start
        over = elapsed > self.manifest.latency_budget_ms
        record = {"request_id": request_id, "kind": "preview_query",
                  "elapsed_ms": elapsed, "budget_ms": self.manifest.latency_budget_ms,
                  "over_budget": over}
        self.ledger.append("latency", record)
        if over:
            # 超预算记录入帐（不剔除请求），并抛出 E_LATENCY 供上层标注。
            self.ledger.append("latency_violation", {
                "code": "E_LATENCY", "request_id": request_id,
                "measured_ms": elapsed, "budget_ms": self.manifest.latency_budget_ms})
            raise LatencyError(
                f"查询 {request_id} 延迟 {elapsed}ms 超出预算 "
                f"{self.manifest.latency_budget_ms}ms",
                field=f"query[{request_id}].elapsed_ms",
                basis=f"measured={elapsed} > budget={self.manifest.latency_budget_ms}",
                extra={"request_id": request_id, "measured_ms": elapsed,
                       "budget_ms": self.manifest.latency_budget_ms})
        return {"request_id": request_id, "elapsed_ms": elapsed, "nodes": snapshot}

    # ---------- 账本权威重算 ----------
    def authoritative_recompute(self, version_number, request_id=None,
                                timing_tolerance_ms=DEFAULT_TIMING_TOLERANCE_MS):
        """从账本逐条重放编译/下发/回执，重算聚合摘要与耗时统计。"""
        request_id = request_id or self._next_request_id("recomp")
        start = mono_ms()
        state = self.ledger.snapshot_state()
        cur = self.manifest.version(version_number)
        slices = build_node_slices(cur, self.registry)
        authoritative = {}
        replay_findings = []
        for node_id in sorted(slices):
            node = self.registry.nodes[node_id]
            items = slices[node_id]
            summary, digest = aggregate_slice(
                node, version_number, items, self.sample_decl)
            task_id = f"v{version_number}:{node_id}"
            receipt = state["receipts"].get(task_id)
            if receipt is None:
                replay_findings.append({"task_id": task_id, "node_id": node_id,
                                        "reason": "账本无回执（未完成/未知，需重跑）"})
                continue
            # 与已提交回执逐项比对（恢复后重放校验的核心）。
            if receipt["summary_digest"] != digest:
                raise ReplayError(
                    f"重放聚合摘要不一致 task={task_id}",
                    field=f"{task_id}.summary_digest",
                    basis=f"ledger={receipt['summary_digest']} replay={digest}")
            if receipt["summary"] != summary:
                raise ReplayError(
                    f"重放聚合摘要内容不一致 task={task_id}",
                    field=f"{task_id}.summary",
                    basis="规范化摘要对象必须相等")
            authoritative[task_id] = {
                "node_id": node_id, "version": version_number,
                "summary": summary, "summary_digest": digest,
                "elapsed_ms": receipt["elapsed_ms"]}
        elapsed = mono_ms() - start
        over = elapsed > self.manifest.latency_budget_ms
        self.ledger.append("latency", {"request_id": request_id,
                                       "kind": "authoritative_recompute",
                                       "elapsed_ms": elapsed,
                                       "budget_ms": self.manifest.latency_budget_ms,
                                       "over_budget": over})
        if over:
            self.ledger.append("latency_violation", {
                "code": "E_LATENCY", "request_id": request_id,
                "measured_ms": elapsed,
                "budget_ms": self.manifest.latency_budget_ms})
        return {"request_id": request_id, "recompute_elapsed_ms": elapsed,
                "authoritative": authoritative, "open_tasks": replay_findings}

    def compare_paths(self, version_number,
                      timing_tolerance_ms=DEFAULT_TIMING_TOLERANCE_MS):
        """快速路径（增量缓存）vs 权威路径，逐字节/误差界判定，首个不一致即 E_PREVIEW。"""
        recompute = self.authoritative_recompute(version_number)
        for task_id in sorted(recompute["authoritative"]):
            auth = recompute["authoritative"][task_id]
            node_id = auth["node_id"]
            view = self.cache[node_id]
            if view.config_version != auth["version"]:
                raise PreviewError(
                    f"节点 {node_id} 版本不一致",
                    field=f"{task_id}.config_version",
                    basis=f"preview={view.config_version} authoritative={auth['version']}",
                    extra={"node_id": node_id, "version": auth["version"]})
            if view.summary_digest != auth["summary_digest"]:
                raise PreviewError(
                    f"节点 {node_id} 聚合摘要不一致",
                    field=f"{task_id}.summary_digest",
                    basis=f"preview={view.summary_digest} "
                          f"authoritative={auth['summary_digest']}",
                    extra={"node_id": node_id, "version": auth["version"]})
            if view.summary != auth["summary"]:
                raise PreviewError(
                    f"节点 {node_id} 聚合摘要内容不一致",
                    field=f"{task_id}.summary",
                    extra={"node_id": node_id, "version": auth["version"]})
            if abs(view.last_elapsed_ms - auth["elapsed_ms"]) > timing_tolerance_ms:
                raise PreviewError(
                    f"节点 {node_id} 耗时统计超出误差界",
                    field=f"{task_id}.elapsed_ms",
                    basis=f"|preview={view.last_elapsed_ms} - "
                          f"authoritative={auth['elapsed_ms']}| > "
                          f"tolerance={timing_tolerance_ms}",
                    extra={"node_id": node_id, "version": auth["version"]})
        return {"match": True, "checked": sorted(recompute["authoritative"]),
                "timing_tolerance_ms": timing_tolerance_ms}

    # ---------- 延迟统计 ----------
    def latency_report(self):
        """按节点与整体统计均值/P95/最大值的定点表示（含超预算请求，不剔除）。"""
        records = [p for _, _, p in self.ledger.events("latency")]
        overall = fixedpoint.latency_stats([r["elapsed_ms"] for r in records])
        per_node = {}
        for r in records:
            node = r.get("node_id", "_query")
            per_node.setdefault(node, []).append(r["elapsed_ms"])
        per_node_stats = {node: fixedpoint.latency_stats(vals)
                          for node, vals in sorted(per_node.items())}
        violations = [p for _, _, p in self.ledger.events("latency_violation")]
        return {"clock": "time.monotonic()*1000 整数毫秒",
                "truth_source": "ledger.events(kind=latency)",
                "overall": overall, "per_node": per_node_stats,
                "violations": violations, "record_count": len(records)}

    # ---------- 强制终止后的恢复与重放 ----------
    def recover(self):
        """重启后仅依据已完整提交的账本事件恢复缓存与判定未知任务。

        未完整提交（无 task_outcome）的任务标记 UNKNOWN 并要求重跑；
        该方法不生成任何“退出报告”，只返回恢复结论。
        """
        state = self.ledger.snapshot_state()
        self.ledger.assert_intact()
        # 由账本重建增量缓存（缓存不是事实来源，账本才是）。
        rebuilt = {nid: NodeView(nid) for nid in sorted(self.registry.nodes)}
        unknown, replay_mismatches = [], []
        receipts = state["receipts"]
        outcomes = state["task_outcomes"]
        for task_id in sorted(outcomes):
            outcome = outcomes[task_id]
            node_id = outcome["node_id"]
            view = rebuilt[node_id]
            view.last_task_id = task_id
            view.last_outcome = outcome["outcome"]
            if outcome["outcome"] == "SUCCESS" and task_id in receipts:
                receipt = receipts[task_id]
                view.config_version = receipt["version"]
                view.summary = receipt["summary"]
                view.summary_digest = receipt["summary_digest"]
                view.last_elapsed_ms = receipt["elapsed_ms"]
                view.progress = "CURRENT"
                # 重放已完成任务并逐项比对，首个不一致即 E_REPLAY。
                self._replay_one(task_id, receipt, replay_mismatches)
            else:
                view.progress = "FAILED"
        # 游标中声明过、但没有结局事件的任务 = 崩溃时未完整提交。
        for cursor_name in sorted(state["cursors"]):
            for task_id in state["cursors"][cursor_name].get("task_ids", []):
                if task_id not in outcomes:
                    unknown.append(task_id)
        with self.cache_lock:
            self.cache = rebuilt
        return {"format_version": LEDGER_FORMAT_VERSION,
                "ledger_replaced": self.ledger.replaced,
                "backup_path": self.ledger.backup_path,
                "restored_nodes": sorted(
                    nid for nid, v in rebuilt.items() if v.progress == "CURRENT"),
                "unknown_tasks": sorted(set(unknown)),
                "replay_mismatches": replay_mismatches}

    def _replay_one(self, task_id, receipt, mismatches):
        try:
            version_number = receipt["version"]
            cur = self.manifest.version(version_number)
            slices = build_node_slices(cur, self.registry)
            node_id = receipt["node_id"]
            summary, digest = aggregate_slice(
                self.registry.nodes[node_id], version_number,
                slices[node_id], self.sample_decl)
        except Exception as exc:  # 重算本身失败也作为不一致上报
            mismatches.append({"task_id": task_id, "field": "recompute",
                               "basis": str(exc)})
            raise ReplayError(f"任务 {task_id} 重算失败",
                              field=f"{task_id}.recompute", basis=str(exc))
        checks = (
            ("summary_digest", receipt["summary_digest"], digest),
            ("summary", receipt["summary"], summary),
            ("config_version", receipt["config_version"], version_number),
        )
        for field_name, ledger_value, replay_value in checks:
            if ledger_value != replay_value:
                raise ReplayError(
                    f"恢复重放不一致 task={task_id}",
                    field=f"{task_id}.{field_name}",
                    basis=f"ledger={json.dumps(ledger_value, sort_keys=True)[:200]} "
                          f"replay={json.dumps(replay_value, sort_keys=True)[:200]}")
        mismatches.append({"task_id": task_id, "field": "all", "match": True})

    def rerun_unknown_tasks(self, version_number):
        """对恢复中标记为未知的任务重新执行完整下发（仅重跑，不伪造结果）。"""
        state = self.ledger.snapshot_state()
        unknown = [
            tid for tid in state["cursors"].get(
                f"v{version_number}:cursor", {}).get("task_ids", [])
            if tid not in state["task_outcomes"]
        ]
        if not unknown:
            return {"rerun": []}
        cur = self.manifest.version(version_number)
        slices = build_node_slices(cur, self.registry)
        only_nodes = {task_id.split(":", 1)[1] for task_id in unknown}
        tasks, task_specs = self._build_dispatch(
            version_number, slices, only_nodes=only_nodes)
        scheduler = Scheduler(self.max_global, self.per_node_quota)
        results = scheduler.run(tasks)
        self._record_results(version_number, results, task_specs)
        return {"rerun": sorted(unknown), "results": results}

    def close(self):
        self.stop_nodes()
        self.ledger.close()

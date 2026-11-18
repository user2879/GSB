"""双重预算约束的并行就绪指令调度器。

预算模型
--------
- 全局槽位：``threading.Semaphore(global_slots)``，任一时刻真实执行的
  指令数不超过 global_slots。
- 每阶段配额：``Semaphore(stage.quota)``，任一时刻该阶段真实执行数不超配额。
- 缓存命中不占用执行槽位（不启动子进程），在调度线程内联完成，因此
  「命中层未被执行」可被审计直接验证。
- 槽位/配额耗尽时就绪指令在确定性就绪队列中排队，记录单调时钟的
  等待开始/结束毫秒；预算只允许排队等待，绝不允许绕过超限。

确定性
------
就绪队列按 (阶段拓扑序号, 指令 seq) 的全序弹出；层摘要只依赖清单、输入
文件与前驱摘要，故并行交错与就绪顺序不改变任何层摘要与最终结果。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Queue

from . import digest, layer as layer_mod
from .cache import CacheIndex
from .errors import (
    E_COMMAND, E_TIMEOUT, OK, STATUS_COMMITTED, STATUS_SKIPPED,
)
from .executor import run_command
from .ledger import Ledger
from .manifest import Manifest, Instruction, Stage


def monotonic_ms() -> int:
    return int(round(time.monotonic() * 1000))


def wall_ms() -> int:
    return int(round(time.time() * 1000))


@dataclass
class Node:
    uid: str
    stage: str
    stage_topo: int
    seq: int
    instruction: Instruction
    predecessors: set[str] = field(default_factory=set)
    successors: set[str] = field(default_factory=set)
    is_stage_root: bool = False


@dataclass
class NodeRecord:
    node: Node
    status: str = "pending"          # pending/running/committed/skipped
    result_code: str = OK
    cache_hit: bool = False
    precursor: str | None = None
    layer_digest: str | None = None
    result_digest: str | None = None
    exit_code: int | None = None
    stdout_digest: str | None = None
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    wait_elapsed_ms: int | None = None
    elapsed_ms: int | None = None
    skipped_trigger: str | None = None
    timeout_residual: str | None = None
    timeout_mechanism: str | None = None
    # 槽位占用时间线（真实执行才有），用于峰值与配额审计
    slot_start_ms: int | None = None
    slot_end_ms: int | None = None

    def as_dict(self) -> dict:
        return {
            "node_uid": self.node.uid,
            "stage": self.node.stage,
            "seq": self.node.seq,
            "id": self.node.instruction.id,
            "type": self.node.instruction.type,
            "params": self.node.instruction.params,
            "status": self.status,
            "result_code": self.result_code,
            "cache_hit": self.cache_hit,
            "precursor": self.precursor,
            "layer_digest": self.layer_digest,
            "result_digest": self.result_digest,
            "exit_code": self.exit_code,
            "stdout_digest": self.stdout_digest,
            "wait_elapsed_ms": self.wait_elapsed_ms,
            "elapsed_ms": self.elapsed_ms,
            "skipped_trigger": self.skipped_trigger,
            "timeout_residual": self.timeout_residual,
            "timeout_mechanism": self.timeout_mechanism,
        }


def build_dag(manifest: Manifest) -> dict[str, Node]:
    """构建跨阶段 DAG。

    - 阶段内：显式 after 建立边；无 after 的指令为该阶段根。
    - 跨阶段：每个阶段根依赖其所有上游阶段根（确保上游阶段先产出）。
    """
    topo_index = {name: i for i, name in enumerate(manifest.topo_order)}
    nodes: dict[str, Node] = {}
    stage_nodes: dict[str, list[Node]] = {}
    stage_roots: dict[str, list[Node]] = {}

    for stage_name in manifest.topo_order:
        stage = manifest.stage_by_name[stage_name]
        local: list[Node] = []
        id_to_uid: dict[str, str] = {}
        for ins in stage.instructions:
            uid = f"{stage_name}#{ins.seq}:{ins.id}"
            node = Node(
                uid=uid, stage=stage_name, stage_topo=topo_index[stage_name],
                seq=ins.seq, instruction=ins,
            )
            nodes[uid] = node
            local.append(node)
            id_to_uid[ins.id] = uid
        stage_nodes[stage_name] = local
        for node in local:
            for ref in node.instruction.after:
                pred_uid = id_to_uid[ref]
                node.predecessors.add(pred_uid)
                nodes[pred_uid].successors.add(node.uid)
        roots = [n for n in local if not n.instruction.after]
        for node in roots:
            node.is_stage_root = True
        stage_roots[stage_name] = roots

    # 跨阶段边：本阶段根 -> 上游阶段根
    for stage_name in manifest.topo_order:
        stage = manifest.stage_by_name[stage_name]
        upstream_roots: list[Node] = []
        for dep in stage.depends_on:
            upstream_roots.extend(stage_roots[dep])
        for root in stage_roots[stage_name]:
            for up in upstream_roots:
                root.predecessors.add(up.uid)
                up.successors.add(root.uid)

    _assert_acyclic(nodes)
    return nodes


def _assert_acyclic(nodes: dict[str, Node]) -> None:
    indegree = {uid: len(n.predecessors) for uid, n in nodes.items()}
    ready = sorted(
        (uid for uid, d in indegree.items() if d == 0),
        key=lambda u: (nodes[u].stage_topo, nodes[u].seq),
    )
    seen = 0
    while ready:
        uid = ready.pop(0)
        seen += 1
        for succ in sorted(
            nodes[uid].successors,
            key=lambda u: (nodes[u].stage_topo, nodes[u].seq),
        ):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort(key=lambda u: (nodes[u].stage_topo, nodes[u].seq))
    if seen != len(nodes):
        # 阶段依赖无环已在清单校验保证；此处为防御性检查
        raise RuntimeError("调度 DAG 检测到环（不应发生，清单已校验）")


class Scheduler:
    def __init__(self, manifest: Manifest, cache: CacheIndex, ledger: Ledger,
                 build_id: str, crash_at: str | None = None):
        self.manifest = manifest
        self.cache = cache
        self.ledger = ledger
        self.build_id = build_id
        self.nodes = build_dag(manifest)
        self.records = {uid: NodeRecord(node) for uid, node in self.nodes.items()}
        self.global_slots = threading.Semaphore(manifest.global_slots)
        self.stage_semaphores = {
            s.name: threading.Semaphore(s.quota) for s in manifest.stages
        }
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self.remaining = set(self.nodes)
        # 环境变量累积（阶段内按 seq 顺序对后续 run 生效）
        self.stage_env: dict[str, dict[str, str]] = {
            s.name: {} for s in manifest.stages
        }
        self.peak_global = 0
        self.running_global = 0
        self.peak_stage = {s.name: 0 for s in manifest.stages}
        self.running_stage = {s.name: 0 for s in manifest.stages}
        self.work_queue: Queue = Queue()
        self.crash_at = crash_at  # 形如 "stage/seq"，到达该指令执行前强制退出
        self.aborted = False
        self.phase = "initial"  # initial / resume，写入事件 detail 供审计分段

    # ---------- 主流程 ----------
    def run(self) -> dict[str, NodeRecord]:
        self.ledger.event(self.build_id, "build_start", wall_ms())
        worker_count = self.manifest.global_slots
        workers = [
            threading.Thread(target=self._worker, name=f"lb-worker-{i}",
                             daemon=True)
            for i in range(worker_count)
        ]
        for worker in workers:
            worker.start()

        with self._cond:
            self._enqueue_ready_locked()
            while self.remaining:
                self._cond.wait(timeout=0.05)
                self._enqueue_ready_locked()
        for worker in workers:
            self.work_queue.put(None)
        for worker in workers:
            worker.join()

        self.ledger.event(self.build_id, "build_end", wall_ms())
        return self.records

    def _enqueue_ready_locked(self) -> None:
        """把所有前驱已完成且尚未入队的就绪节点按全序放入工作队列。"""
        ready = [
            uid for uid in self.remaining
            if self.records[uid].status == "pending"
            and all(self.records[p].status in ("committed", "skipped")
                    for p in self.nodes[uid].predecessors)
        ]
        ready.sort(key=lambda u: (self.nodes[u].stage_topo,
                                  self.nodes[u].seq))
        for uid in ready:
            # 标记为 queued，避免重复入队；等待计时从入队开始
            record = self.records[uid]
            record.status = "queued"
            record.wait_start_ms = monotonic_ms()
            self.remaining.discard(uid)
            self.work_queue.put(uid)
            self.ledger.event(self.build_id, "queued", wall_ms(), uid)

    def _worker(self) -> None:
        while True:
            uid = self.work_queue.get()
            if uid is None:
                return
            try:
                self._process(uid)
            finally:
                with self._cond:
                    self._cond.notify_all()

    def _predecessor_layers(self, node: Node) -> dict[str, str]:
        layers = {}
        for pred_uid in sorted(node.predecessors):
            pred_record = self.records[pred_uid]
            if pred_record.layer_digest is not None:
                layers[pred_uid] = pred_record.layer_digest
        return layers

    def _process(self, uid: str) -> None:
        node = self.nodes[uid]
        record = self.records[uid]
        record.wait_end_ms = monotonic_ms()
        record.wait_elapsed_ms = max(
            0, record.wait_end_ms - (record.wait_start_ms or record.wait_end_ms)
        )

        # 上游若为 skipped，则本指令传播跳过（不再申请槽位）
        skipped_pred = self._skipped_predecessor(node)
        if skipped_pred is not None:
            self._apply_skip(record, node, skipped_pred)
            return

        precursor, inputs = layer_mod.precursor_digest(
            node.instruction, self.manifest, self._predecessor_layers(node)
        )
        record.precursor = precursor

        # 缓存判定在申请执行槽位之前；命中内联完成、不启动子进程、不占槽
        cached = self.cache.lookup(precursor)
        if cached is not None:
            record.cache_hit = True
            record.layer_digest = cached.layer_digest
            record.result_digest = cached.result_digest
            record.exit_code = cached.exit_code
            record.stdout_digest = cached.stdout_digest
            record.elapsed_ms = 0
            record.status = STATUS_COMMITTED
            record.result_code = OK
            self.cache.save()
            self.ledger.event(self.build_id, "cache_hit", wall_ms(), uid,
                              {"layer_digest": cached.layer_digest})
            self._commit(record, node)
            return

        # 未命中：申请双重预算（耗尽则阻塞排队，绝不绕过）
        if node.instruction.type == "run":
            self._acquire_slots(node, record)
            if self.crash_at and self._crash_matches(node):
                self._crash(node, record)  # 该调用不会返回
            command_result = self._execute(node, record)
            self._release_slots(node)
            self._handle_run_result(record, node, precursor, inputs,
                                    command_result)
        else:
            self._handle_non_run(record, node, precursor, inputs)

    def _crash_matches(self, node: Node) -> bool:
        target = self.crash_at or ""
        return target == f"{node.stage}/{node.seq}"

    def _crash(self, node: Node, record: NodeRecord) -> None:
        """模拟强制终止：running 已落库但指令未提交，随后硬退出进程。"""
        record.status = "running"
        record.slot_start_ms = monotonic_ms()
        self.ledger.mark_running(self.build_id, node, wall_ms())
        self.ledger.event(self.build_id, "injected_crash", wall_ms(),
                          node.uid)
        # os._exit 不抛异常、不执行 finally，最接近 SIGKILL / 强制结束
        os_exit = __import__("os")._exit
        os_exit(137)

    # ---------- 预算 ----------
    def _acquire_slots(self, node: Node, record: NodeRecord) -> None:
        self.ledger.event(self.build_id, "slot_wait", wall_ms(), node.uid,
                          {"phase": self.phase})
        self.global_slots.acquire()
        self.stage_semaphores[node.stage].acquire()
        with self._cond:
            self.running_global += 1
            self.running_stage[node.stage] += 1
            self.peak_global = max(self.peak_global, self.running_global)
            self.peak_stage[node.stage] = max(
                self.peak_stage[node.stage], self.running_stage[node.stage]
            )
            record.slot_start_ms = monotonic_ms()
        self.ledger.event(
            self.build_id, "slot_acquired", wall_ms(), node.uid,
            {"running_global": self.running_global,
             "running_stage": self.running_stage[node.stage],
             "phase": self.phase},
        )

    def _release_slots(self, node: Node) -> None:
        with self._cond:
            self.records[node.uid].slot_end_ms = monotonic_ms()
            self.running_global -= 1
            self.running_stage[node.stage] -= 1
        self.stage_semaphores[node.stage].release()
        self.global_slots.release()
        self.ledger.event(self.build_id, "slot_released", wall_ms(), node.uid,
                          {"phase": self.phase})

    # ---------- 执行 ----------
    def _execute(self, node: Node, record: NodeRecord):
        record.status = "running"
        self.ledger.mark_running(self.build_id, node, wall_ms())
        timeout_ms = node.instruction.params.get(
            "timeout_ms", self.manifest.default_timeout_ms
        )
        env = dict(sorted(self.stage_env[node.stage].items()))
        result = run_command(
            node.instruction.params["command"], timeout_ms,
            env=env, cwd=self.manifest.workspace,
        )
        record.elapsed_ms = result.elapsed_ms
        record.timeout_residual = result.residual
        record.timeout_mechanism = result.mechanism
        return result

    def _handle_run_result(self, record, node, precursor, inputs, result) -> None:
        if result.timed_out:
            record.status = STATUS_COMMITTED
            record.result_code = E_TIMEOUT
            record.exit_code = result.exit_code
            record.stdout_digest = result.stdout_digest
            record.result_digest = layer_mod.command_result_digest(
                result.exit_code, result.stdout_digest
            )
            record.layer_digest = layer_mod.layer_digest_run(
                precursor, record.result_digest
            )
            self.ledger.clear_running(self.build_id, node.uid)
            self._commit(record, node)
            return

        record.exit_code = result.exit_code
        record.stdout_digest = result.stdout_digest
        record.result_digest = layer_mod.command_result_digest(
            result.exit_code, result.stdout_digest
        )
        record.layer_digest = layer_mod.layer_digest_run(
            precursor, record.result_digest
        )
        if result.exit_code != 0:
            record.status = STATUS_COMMITTED
            record.result_code = E_COMMAND
        else:
            record.status = STATUS_COMMITTED
            record.result_code = OK
        entry, evictions = self.cache.insert(
            precursor, record.layer_digest, inputs, record.result_digest,
            result.exit_code, result.stdout_digest,
        )
        for ev in evictions:
            self.ledger.record_eviction(self.build_id, ev)
        self.cache.save()
        self.ledger.clear_running(self.build_id, node.uid)
        self._commit(record, node)

    def _handle_non_run(self, record, node, precursor, inputs) -> None:
        record.layer_digest = layer_mod.layer_digest_non_run(precursor)
        record.result_digest = layer_mod.NO_COMMAND_RESULT
        record.status = STATUS_COMMITTED
        record.result_code = OK
        record.elapsed_ms = 0
        # 非执行层也进入缓存（容量预算同样约束），保证索引完整
        _, evictions = self.cache.insert(
            precursor, record.layer_digest, inputs,
            layer_mod.NO_COMMAND_RESULT, None, None,
        )
        for ev in evictions:
            self.ledger.record_eviction(self.build_id, ev)
        self.cache.save()
        if node.instruction.type == "env":
            self.stage_env[node.stage][node.instruction.params["name"]] = \
                node.instruction.params["value"]
        self._commit(record, node)

    def _skipped_predecessor(self, node: Node) -> str | None:
        for pred_uid in sorted(node.predecessors):
            if self.records[pred_uid].status == STATUS_SKIPPED:
                return pred_uid
            if self.records[pred_uid].result_code in (E_COMMAND, E_TIMEOUT):
                return pred_uid
        return None

    def _apply_skip(self, record: NodeRecord, node: Node,
                    trigger_uid: str) -> None:
        trigger = self.records[trigger_uid]
        record.status = STATUS_SKIPPED
        record.result_code = trigger.result_code
        record.skipped_trigger = trigger_uid
        record.cache_hit = False
        record.layer_digest = None
        record.elapsed_ms = 0
        self.ledger.event(
            self.build_id, "skipped", wall_ms(), node.uid,
            {"trigger": trigger_uid, "code": trigger.result_code},
        )
        self._commit(record, node)

    def _commit(self, record: NodeRecord, node: Node) -> None:
        payload = record.as_dict()
        self.ledger.commit_instruction(self.build_id, node, payload)
        self.ledger.event(
            self.build_id, "committed", wall_ms(), node.uid,
            {"status": record.status, "code": record.result_code,
             "cache_hit": record.cache_hit},
        )

    # ---------- 强制终止恢复续建 ----------
    def prime_from_ledger(self, rows) -> set[str]:
        """用账本中完整提交的记录预填调度状态，返回仍需重跑的 uid 集合。

        仅承认 status=committed/skipped 的终态记录；running 残留不预填，
        其节点保持 pending（语义等同 unknown，要求重跑）。
        """
        committed: set[str] = set()
        for row in rows:
            uid = row["node_uid"]
            if uid not in self.records:
                continue
            record = self.records[uid]
            record.status = row["status"]
            record.result_code = row["result_code"]
            record.cache_hit = bool(row["cache_hit"])
            record.precursor = row["precursor"]
            record.layer_digest = row["layer_digest"]
            record.result_digest = row["result_digest"]
            record.exit_code = row["exit_code"]
            record.stdout_digest = row["stdout_digest"]
            record.skipped_trigger = row["skipped_trigger"]
            record.timeout_residual = row["timeout_residual"]
            record.timeout_mechanism = row["timeout_mechanism"]
            if row["status"] in ("committed", "skipped"):
                committed.add(uid)
                # 重建阶段环境，保证续建时后续 run 的环境一致
                node = self.nodes[uid]
                if node.instruction.type == "env" and row["status"] == "committed":
                    self.stage_env[node.stage][
                        node.instruction.params["name"]
                    ] = node.instruction.params["value"]
        # 续建段独立计数：清零占用统计，仅反映本次重跑的峰值
        self.peak_global = 0
        self.running_global = 0
        self.peak_stage = {name: 0 for name in self.peak_stage}
        self.running_stage = {name: 0 for name in self.running_stage}
        for row in rows:
            self.ledger.clear_running(self.build_id, row["node_uid"])
        rerun = set(self.nodes) - committed
        # 仅保留需要重跑的节点参与调度
        self.remaining = set(rerun)
        return rerun


    def resume(self) -> dict[str, NodeRecord]:
        """续建：已提交节点不再调度，仅推进 unknown 及其下游。"""
        self.phase = "resume"
        self.ledger.event(self.build_id, "resume_start", wall_ms())
        worker_count = self.manifest.global_slots
        workers = [
            threading.Thread(target=self._worker, name=f"lb-resume-{i}",
                             daemon=True)
            for i in range(worker_count)
        ]
        for worker in workers:
            worker.start()
        with self._cond:
            self._enqueue_ready_locked()
            while self.remaining:
                self._cond.wait(timeout=0.05)
                self._enqueue_ready_locked()
        for worker in workers:
            self.work_queue.put(None)
        for worker in workers:
            worker.join()
        self.ledger.event(self.build_id, "resume_end", wall_ms())
        return self.records

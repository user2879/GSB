"""平台门面：编排归一、节点、调度、账本、预览与权威重算。"""

import json
import os

from .canon import canon
from .digest import compute_summary, digest_of, effective_value
from .errors import (ConvergeError, FormatError, LatencyError, PreviewError,
                     ReplayError)
from .latency import LatencyRecorder
from .ledger import Ledger
from .manifest import decode_manifest, encode_manifest
from .normalize import normalize_directory
from .orchestrator import NodeManager
from .registry import load_registry
from .samples import generate_samples, load_data
from .scheduler import Scheduler, SliceTask

DEFAULT_BUDGETS_MS = {"ui_query": 50, "recompute": 1000}


def _atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class Platform:
    def __init__(self, workdir):
        self.workdir = workdir
        self.sample_dir = os.path.join(workdir, "samples")
        self.manifest_dir = os.path.join(self.sample_dir, "manifests")
        self.data_path = os.path.join(self.sample_dir, "data.json")
        self.registry_path = os.path.join(self.sample_dir, "registry.json")
        self.out_dir = os.path.join(workdir, "out")
        self.state_dir = os.path.join(self.out_dir, "node_state")
        self.ledger_path = os.path.join(self.out_dir, "ledger.db")
        self.series = None
        self.specs = None
        self.ledger = None
        self.nodes = None
        self.scheduler = None
        self.recorder = LatencyRecorder()
        self.current_manifest = None
        self.current_manifest_name = None
        # 预览缓存：version -> {(node,key): {"digest","summary","stale"}}
        self.preview = {}
        self.preview_version = 0
        self.preview_archive = {}

    # ---- 样本与归一 ----
    def gen_samples(self):
        return generate_samples(self.sample_dir)

    def normalize(self, hook=None):
        return normalize_directory(self.manifest_dir, hook=hook)

    # ---- 生命周期 ----
    def open(self):
        os.makedirs(self.state_dir, exist_ok=True)
        self.series = load_data(self.data_path)
        self.specs = load_registry(self.registry_path)
        self.ledger = Ledger(self.ledger_path)
        self.nodes = NodeManager(self.specs, self.data_path, self.state_dir)
        return self.recover()

    def close(self):
        if self.nodes:
            self.nodes.stop_all()
        if self.ledger:
            self.ledger.close()

    def recover(self):
        unknown = self.ledger.recover()
        return {"unknown_batches": unknown,
                "note": "运行中未提交的下发已标记为未知，须重跑；不生成退出报告"}

    def start_nodes(self):
        return self.nodes.start()

    def restart_node(self, node_id):
        return self.nodes.restart(node_id)

    # ---- 清单 ----
    def load_manifest(self, name):
        path = os.path.join(self.manifest_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            manifest = decode_manifest(fh.read())
        self.current_manifest = manifest
        self.current_manifest_name = name
        return manifest

    def _key_versions(self):
        return json.loads(self.ledger.get_meta("key_versions", "{}"))

    def _resolve_targets(self, targets, path):
        if targets == ["*"]:
            return sorted(self.specs)
        nodes = []
        for t in targets:
            if t not in self.specs:
                raise FormatError(f"目标节点不存在: {t!r}", path=path,
                                  source_version="v3")
            nodes.append(t)
        return sorted(nodes)

    # ---- 下发 ----
    def _build_tasks(self, manifest):
        key_versions = self._key_versions()
        mv = manifest.manifest_version
        context = {e.key: e.value for e in manifest.entries}
        derived = [{"key": d.key, "op": d.op, "inputs": d.inputs}
                   for d in manifest.derived]
        units = [(e.key, e.value, e.targets, e.priority, f"entries[{i}]")
                 for i, e in enumerate(manifest.entries)]
        units += [(d.key, None, d.targets, d.priority, f"derived[{i}]")
                  for i, d in enumerate(manifest.derived)]
        tasks = []
        records = []
        for key, value, targets, priority, path in units:
            kv = key_versions.setdefault(key, mv)
            for node in self._resolve_targets(targets, f"{path}.targets"):
                payload = {"op": "apply", "manifest_version": mv, "key": key,
                           "key_version": kv, "value": value,
                           "context": context, "derived": derived}
                eff = effective_value(key, value, context, derived)
                profile = self.specs[node].profile
                summary = compute_summary(mv, profile, [[key, eff, kv]],
                                          self.series)
                task = SliceTask(priority=priority, key=key, node=node,
                                 payload=payload,
                                 timeout_ms=manifest.strategy.timeout_ms,
                                 max_retries=manifest.strategy.max_retries)
                tasks.append(task)
                records.append({"node": node, "key": key,
                                "priority": priority,
                                "payload_json": canon(payload),
                                "expected_summary": summary,
                                "expected_digest": digest_of(summary)})
        return tasks, records, key_versions

    def dispatch(self, manifest=None):
        manifest = manifest or self.current_manifest
        if manifest is None:
            raise FormatError("尚未加载清单", path="$", source_version="v3")
        mv = manifest.manifest_version
        tasks, records, key_versions = self._build_tasks(manifest)
        batch_id = self.ledger.begin_batch(mv, encode_manifest(manifest),
                                           records)
        expected = {(r["node"], r["key"]): (r["expected_summary"],
                                            r["expected_digest"])
                    for r in records}
        self.scheduler = Scheduler(manifest.strategy.global_concurrency,
                                   manifest.strategy.node_quota)
        results = self.scheduler.execute(tasks, self.nodes.dispatch)

        deviations = []
        commit_rows = []
        for res in results:
            node, key = res.task.node, res.task.key
            if not res.ok:
                deviations.append({"node": node, "key": key,
                                   "expected": expected[(node, key)][1],
                                   "actual": None,
                                   "error": res.error})
                commit_rows.append({"node": node, "key": key,
                                    "status": "failed",
                                    "attempts": res.attempts,
                                    "receipt": None})
                continue
            exp_summary, exp_digest = expected[(node, key)]
            rec = res.receipt
            actual_summary = rec.get("summary")
            match = rec["digest"] == exp_digest and (
                rec["view"] == "s1" or actual_summary == exp_summary)
            if not match:
                deviations.append({"node": node, "key": key,
                                   "expected": exp_digest,
                                   "actual": rec["digest"]})
            commit_rows.append({"node": node, "key": key,
                                "status": "ok" if match else "mismatch",
                                "attempts": res.attempts, "receipt": rec})
        deviations.sort(key=lambda d: (d["node"], d["key"]))
        quota = self.scheduler.quota_stats()
        self.ledger.commit_batch(batch_id, commit_rows, quota)
        self.ledger.set_meta("last_manifest_version", str(mv))
        self.ledger.set_meta("key_versions", json.dumps(key_versions,
                                                        sort_keys=True))
        self._replace_preview(mv, commit_rows)
        report = {"batch_id": batch_id, "manifest_version": mv,
                  "slices": len(results), "ok": sum(1 for r in results
                                                    if r.ok),
                  "start_order": self.scheduler.start_order(),
                  "quota": quota, "deviations": deviations}
        if deviations:
            report["error"] = ConvergeError(deviations).to_dict()
        return report

    def _replace_preview(self, mv, commit_rows):
        """整体替换当前视图；旧版本视图归档为只读。"""
        if self.preview_version and self.preview_version != mv:
            self.preview_archive[self.preview_version] = self.preview
        if self.preview_version != mv:
            self.preview = {}
            self.preview_version = mv
        for row in commit_rows:
            if row["status"] == "ok":
                rec = row["receipt"]
                self.preview[(row["node"], row["key"])] = {
                    "digest": rec["digest"],
                    "summary": rec.get("summary"),
                    "stale": False}

    # ---- 界面快速视图 ----
    def ui_query(self):
        """界面快速视图查询：只读缓存，记录交互延迟。"""
        with self.recorder.measure("ui_query"):
            receipts = [[node, key, ent["digest"]]
                        for (node, key), ent in sorted(self.preview.items())
                        if not ent["stale"]]
            progress = self.scheduler.snapshot() if self.scheduler else None
            return {"version": self.preview_version,
                    "receipts": receipts,
                    "conclusion": self._conclusion(self.preview_version,
                                                   receipts),
                    "progress": progress}

    @staticmethod
    def _conclusion(version, receipts):
        return canon({"version": version, "receipts": receipts})

    # ---- 权威重算 ----
    def recompute_authoritative(self):
        """按账本逐条重放并重算派生，与快速结论逐字节比较。"""
        with self.recorder.measure("recompute"):
            mv = self.preview_version
            batches = self.ledger.committed_batches(mv)
            for batch_id, _mv, _mj, _st, _sd in batches:
                self.replay_batch(batch_id)
            receipts_all = []
            if batches:
                # 当前视图整体对应最新提交批次，而非跨批次增量合并
                latest = batches[-1][0]
                for node, key, view, summary, digest, elapsed in \
                        self.ledger.load_receipts(latest):
                    receipts_all.append([node, key, digest])
            receipts_all.sort()
            authoritative = self._conclusion(mv, receipts_all)
        preview_conclusion = self.ui_query()["conclusion"]
        if preview_conclusion != authoritative:
            p_map = {tuple(r[:2]): r[2]
                     for r in json.loads(preview_conclusion)["receipts"]}
            a_map = {tuple(r[:2]): r[2]
                     for r in json.loads(authoritative)["receipts"]}
            node = key = None
            for nk in sorted(set(p_map) | set(a_map)):
                if p_map.get(nk) != a_map.get(nk):
                    node, key = nk
                    break
            raise PreviewError(key, node)
        return {"version": mv, "conclusion": authoritative,
                "batches_replayed": len(batches), "match": True}

    def replay_batch(self, batch_id):
        """重放单个批次：重算期望摘要并与账本记录逐字节比对。"""
        for (node, key, _prio, payload_json, exp_summary, exp_digest,
             status, _attempts) in self.ledger.load_slices(batch_id):
            payload = json.loads(payload_json)
            eff = effective_value(payload["key"], payload.get("value"),
                                  payload["context"], payload["derived"])
            profile = self.specs[node].profile
            summary = compute_summary(payload["manifest_version"], profile,
                                      [[key, eff, payload["key_version"]]],
                                      self.series)
            basis = f"batch_id={batch_id}"
            if summary != exp_summary:
                raise ReplayError(key, node, "expected_summary",
                                  exp_summary, summary, basis)
            if digest_of(summary) != exp_digest:
                raise ReplayError(key, node, "expected_digest",
                                  exp_digest, digest_of(summary), basis)
            if status == "ok":
                recs = {(n, k): (d, s) for n, k, _v, s, d, _e in
                        self.ledger.load_receipts(batch_id)}
                got = recs.get((node, key))
                if got is None:
                    raise ReplayError(key, node, "receipt", exp_digest,
                                      None, basis)
                if got[0] != exp_digest:
                    raise ReplayError(key, node, "digest", exp_digest,
                                      got[0], basis)
        return True

    # ---- 编辑与陈旧防护 ----
    def edit_manifest(self, name, new_text):
        new = decode_manifest(new_text)  # E_FORMAT 整体拒绝
        last = int(self.ledger.get_meta("last_manifest_version", "0"))
        new.manifest_version = max(new.manifest_version, last + 1)
        old = self.current_manifest
        changed = []
        if old is not None:
            old_map = {e.key: (canon(e.value), canon(e.targets), e.priority)
                       for e in old.entries}
            old_map.update({d.key: (canon([d.op, d.inputs]),
                                    canon(d.targets), d.priority)
                            for d in old.derived})
            new_map = {e.key: (canon(e.value), canon(e.targets), e.priority)
                       for e in new.entries}
            new_map.update({d.key: (canon([d.op, d.inputs]),
                                    canon(d.targets), d.priority)
                            for d in new.derived})
            changed = sorted(k for k in set(old_map) | set(new_map)
                             if old_map.get(k) != new_map.get(k))
        else:
            changed = sorted(new.all_keys())
        key_versions = self._key_versions()
        for k in changed:
            key_versions[k] = new.manifest_version
        affected = set()
        for unit in list(new.entries) + list(new.derived):
            if unit.key in changed:
                affected.update(self._resolve_targets(unit.targets,
                                                      "targets"))
        if old is not None:
            for unit in list(old.entries) + list(old.derived):
                if unit.key in changed:
                    affected.update(self._resolve_targets(unit.targets,
                                                          "targets"))
        _atomic_write(os.path.join(self.manifest_dir, name),
                      encode_manifest(new))
        self.ledger.supersede_before(new.manifest_version)
        self.ledger.set_meta("last_manifest_version",
                             str(new.manifest_version))
        self.ledger.set_meta("key_versions",
                             json.dumps(key_versions, sort_keys=True))
        for ent in self.preview.values():
            ent["stale"] = True  # 旧版本结论标记为陈旧，不参与当前统计
        self.current_manifest = new
        self.current_manifest_name = name
        return {"manifest_version": new.manifest_version,
                "changed_keys": changed,
                "affected_nodes": sorted(affected)}

    def replacement_report(self, old_version, new_version):
        """整体替换报告：变化的键、受影响节点与旧新摘要（稳定排序）。"""
        old_view = self.preview_archive.get(old_version, {})
        rows = []
        for (node, key), ent in sorted(self.preview.items()):
            if ent["stale"]:
                continue
            old = old_view.get((node, key))
            old_digest = old["digest"] if old else None
            if old_digest != ent["digest"]:
                rows.append({"node": node, "key": key,
                             "old": old_digest, "new": ent["digest"]})
        return {"old_version": old_version, "new_version": new_version,
                "changes": rows}

    # ---- 延迟报告 ----
    def latency_report(self, budgets=None):
        budgets = budgets or DEFAULT_BUDGETS_MS
        events = []
        stats = {}
        for kind in self.recorder.kinds():
            stats[kind] = self.recorder.stats(kind)
            budget = budgets.get(kind)
            if budget is not None and stats[kind]["p95_ms"] > budget:
                events.append(LatencyError(kind, stats[kind]["p95_ms"],
                                           budget).to_dict())
        return {"stats": stats, "events": events}

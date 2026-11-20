"""实验台编排：加载校验 -> 特征抽取 -> 融合打分 -> 产物汇总。

确定性原则：线程池只做纯计算；配额判定、全局预算扣减、缓存驻留/淘汰、
输出排序全部在提交阶段按样本 ID 稳定顺序串行完成，因此任意并发度、
任意预算耗尽时机下，最终产物与单线程串行执行逐字段一致。
"""

import json
import os
import signal
import time

import numpy as np

from .cache import BlockCache
from .errors import (E_BUDGET_EXHAUSTED, E_CONFIG_INVALID, E_CONFIG_MISMATCH,
                     PipelineError)
from .features_image import block_vector, visual_feature_from_blocks
from .features_text import encode_text
from .fusion import fuse
from .loader import load_samples
from .scheduler import (compute_parallel, effective_quota, sample_mac_cost)
from .snapshot import SnapshotManager

STAGE_ORDER = ("load_validate", "schedule", "feature_extract",
               "fusion_score", "summarize")

EXIT_OK = 0
EXIT_CONFIG_INVALID = 2
EXIT_CONFIG_MISMATCH = 3


class StopRequested(Exception):
    """收到可处理退出信号：完成当前阶段提交后以退出码 0 退出。"""


def _atomic_write(path, text):
    tmp_dir = os.path.join(os.path.dirname(path), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, os.path.basename(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp_path, path)


def _write_jsonl(path, records):
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in records)
    _atomic_write(path, text)


def _write_json(path, payload):
    _atomic_write(path, json.dumps(payload, ensure_ascii=False,
                                   sort_keys=True, indent=1) + "\n")


def _sort_failures(failures):
    seen = set()
    unique = []
    for rec in failures:
        key = (rec["sample_id"], rec["stage"], rec["code"], rec["message"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return sorted(unique,
                  key=lambda f: (f["sample_id"],
                                 STAGE_ORDER.index(f["stage"])
                                 if f["stage"] in STAGE_ORDER else 99))


class Pipeline:
    def __init__(self, cfg, plan, samples_dir, output_dir,
                 budget_override=None, workers_override=None,
                 stage_delay_ms=0):
        self.cfg = cfg
        self.plan = plan
        self.samples_dir = samples_dir
        self.output_dir = output_dir
        self.budget = (budget_override if budget_override is not None
                       else cfg.global_mac_budget)
        self.budget_source = ("override" if budget_override is not None
                              else "config")
        self.workers = (workers_override if workers_override is not None
                        else cfg.max_workers)
        self.stage_delay_ms = stage_delay_ms
        self.snap = SnapshotManager(os.path.join(output_dir, "snapshots"))
        self._stop = False
        self.failures = []
        self.pending = []
        self.completion_order = []
        self.resumed = False
        self.input_fingerprint = None
        self.consumed_budget = 0
        self.completed = {"results": [], "failures": [], "ledger": [],
                          "cache_evictions": [], "cache_recomputes": []}
        self.cache = BlockCache(cfg.cache_resident_budget)
        os.makedirs(output_dir, exist_ok=True)

    # ---- 信号处理 ----
    def install_signal_handlers(self):
        def handler(signum, frame):
            self._stop = True
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _stage_point(self):
        if self.stage_delay_ms:
            time.sleep(self.stage_delay_ms / 1000.0)
        if self._stop:
            raise StopRequested()

    # ---- 主流程 ----
    def run(self):
        try:
            return self._run_inner()
        except StopRequested:
            done = {r["sample_id"]
                    for r in self.completed.get("results", [])}
            leftover = [s for s in getattr(self, "_committed_ids", [])
                        if s not in done]
            self.pending = sorted(set(self.pending) | set(leftover))
            self._save_progress()
            self._write_outputs()
            return EXIT_OK

    def _run_inner(self):
        cfg = self.cfg
        manifest = self.snap.load_manifest()
        self.completed = self.snap.load_completed()
        if manifest is not None:
            if (manifest.get("global_version") != cfg.manifest_version
                    or manifest.get("config_hash") != cfg.fingerprint):
                return self._refuse_resume(
                    "manifest version or config hash mismatch")
            self.resumed = True
            self.consumed_budget = int(manifest.get("consumed_budget", 0))
        if self.resumed and "cache_state" in self.completed:
            self.cache.restore(self.completed["cache_state"])
            self.cache.evictions = list(
                self.completed.get("cache_evictions", []))
            self.cache.recomputes = list(
                self.completed.get("cache_recomputes", []))

        # 阶段 1：加载校验
        samples, load_failures, in_fp = load_samples(
            self.samples_dir, cfg, self.plan)
        self.input_fingerprint = in_fp
        if manifest is not None and \
                manifest.get("input_fingerprint") != in_fp:
            return self._refuse_resume("input fingerprint mismatch")
        # 续跑时：上一轮的预算耗尽记录属瞬时状态（本轮按新预算重新判定），
        # 其余失败会被确定性重算，按记录内容去重
        seen = set()
        self.failures = []
        persisted = [r for r in self.completed.get("failures", [])
                     if r["code"] != E_BUDGET_EXHAUSTED]
        for rec in (persisted + load_failures):
            key = (rec["sample_id"], rec["stage"], rec["code"],
                   rec["message"])
            if key not in seen:
                seen.add(key)
                self.failures.append(rec)
        done_ids = {r["sample_id"] for r in self.completed.get("results", [])}
        todo = [s for s in samples if s.sample_id not in done_ids]
        self._commit_stage("load_validate", {
            "validated_samples": [s.sample_id for s in samples],
            "failures": load_failures,
        })
        self._stage_point()

        # 阶段 2：配额判定 + 全局预算分配预演（样本 ID 稳定顺序）；
        # 实际扣减发生在提交阶段逐样本记账，保证中断续跑后余量可继续消耗
        committed, pending = self._allocate(todo)
        self.pending = pending
        self._commit_stage("feature_extract", {
            "committed_samples": [s.sample_id for s in committed],
            "pending_samples": list(pending),
        })
        self._stage_point()

        # 阶段 3：并行计算（纯函数，无副作用）
        results_map, compute_failures, completion_order = compute_parallel(
            committed, cfg, self.plan, self.workers)
        self.completion_order = completion_order
        self.failures += compute_failures
        self._commit_stage("fusion_score", {
            "computed_samples": sorted(results_map),
            "failed_samples": sorted({f["sample_id"]
                                      for f in compute_failures}),
        })
        self._stage_point()

        # 阶段 4：确定性提交（缓存驻留/淘汰审计 + 逐样本核验 + 结果组装）
        sample_by_id = {s.sample_id: s for s in committed}
        self._committed_ids = sorted(results_map)
        results_acc = self.completed.setdefault("results", [])
        for i, sid in enumerate(self._committed_ids):
            record = self._commit_sample(sample_by_id[sid], results_map[sid])
            results_acc.append(record)
            self._charge(sample_by_id[sid])
            if (i + 1) % cfg.snapshot_commit_interval == 0:
                self._save_progress()
            self._stage_point()
        self._save_progress()
        self._write_outputs()
        self._commit_stage("summarize", {
            "total_results": len(self.completed["results"]),
            "total_failures": len(self.failures),
            "consumed_budget": self.consumed_budget,
        })
        return EXIT_OK

    # ---- 预算与配额 ----
    def _allocate(self, todo):
        cfg = self.cfg
        committed, pending = [], []
        projected = self.consumed_budget
        exhausted = False
        for sample in todo:
            cost = sample_mac_cost(sample, cfg)
            quota = effective_quota(sample, cfg, self.plan)
            if not exhausted and cost > quota:
                self.failures.append({
                    "sample_id": sample.sample_id,
                    "stage": "schedule",
                    "code": E_BUDGET_EXHAUSTED,
                    "message": f"sample cost {cost} exceeds per-sample "
                               f"quota {quota}",
                    "injection_id": self._inj_id(sample.sample_id,
                                                 "zero_quota"),
                })
                continue
            if exhausted or cost > self.budget - projected:
                exhausted = True
                pending.append(sample.sample_id)
                self.failures.append({
                    "sample_id": sample.sample_id,
                    "stage": "schedule",
                    "code": E_BUDGET_EXHAUSTED,
                    "message": f"global budget exhausted: cost {cost} > "
                               f"remaining {max(0, self.budget - projected)}",
                    "injection_id": None,
                })
                continue
            committed.append(sample)
            projected += cost
        return committed, pending

    def _charge(self, sample):
        """提交时记账：逐样本扣减全局预算并追加预算扣减序列。"""
        cost = sample_mac_cost(sample, self.cfg)
        self.consumed_budget += cost
        ledger = self.completed.setdefault("ledger", [])
        ledger.append({
            "seq": len(ledger),
            "sample_id": sample.sample_id,
            "cost": cost,
            "remaining_after": self.budget - self.consumed_budget,
        })

    def _inj_id(self, sample_id, inj_type):
        hits = self.plan.for_sample(sample_id, inj_type)
        return hits[0]["id"] if hits else None

    # ---- 确定性提交 ----
    def _commit_sample(self, sample, result):
        cfg = self.cfg
        sid = sample.sample_id
        cache = self.cache

        def vblock(i):
            return block_vector(sample.image, sample.block_size, i, cfg)

        def tchunk(j):
            vec, _ = encode_text(sample.assertions, cfg)
            return vec[j * cfg.block_size:(j + 1) * cfg.block_size]

        # feature_extract 阶段访问
        for i in range(sample.n_blocks):
            cache.access(f"{sid}|v|{i:04d}", lambda i=i: vblock(i))
        for j in range(cfg.n_text_blocks):
            cache.access(f"{sid}|t|{j:04d}", lambda j=j: tchunk(j))
        # fusion_score 阶段再次访问（被淘汰块触发重算审计）
        blocks = [cache.access(f"{sid}|v|{i:04d}", lambda i=i: vblock(i))
                  for i in range(sample.n_blocks)]
        chunks = [cache.access(f"{sid}|t|{j:04d}", lambda j=j: tchunk(j))
                  for j in range(cfg.n_text_blocks)]
        feat_v = visual_feature_from_blocks(blocks, cfg)
        vec_t = np.concatenate(chunks)[:cfg.text_dim] \
            if chunks else np.zeros(cfg.text_dim, dtype=np.int64)
        fused = fuse(feat_v, vec_t, cfg)
        verified = (fused == result["fused"]
                    and bool(np.array_equal(feat_v, result["visual_feature"]))
                    and bool(np.array_equal(vec_t, result["text_feature"])))
        fused = result["fused"]
        return {
            "sample_id": sid,
            "score": int(fused["score"]),
            "decision": fused["decision"],
            "borderline": bool(fused["borderline"]),
            "alpha": {k: int(v) for k, v in fused["alpha"].items()},
            "contributions": {
                m: {k: int(v) for k, v in part.items()}
                for m, part in fused["contributions"].items()
            },
            "visual_feature": [int(x) for x in result["visual_feature"]],
            "text_feature": [int(x) for x in result["text_feature"]],
            "feature_meta": result["visual_meta"],
            "text_info": result["text_info"],
            "commit_verified": verified,
        }

    # ---- 快照与输出 ----
    def _commit_stage(self, stage, artifact):
        self.snap.commit_stage(stage, artifact)

    def _save_progress(self):
        results = list(self.completed.get("results", []))
        state = {
            "results": results,
            "failures": _sort_failures(self.failures),
            "ledger": self.completed.get("ledger", []),
            "cache_evictions": self.cache.evictions,
            "cache_recomputes": self.cache.recomputes,
            "cache_state": self.cache.state(),
        }
        self.snap.save_completed(state)
        self.snap.write_manifest({
            "global_version": self.cfg.manifest_version,
            "config_hash": self.cfg.fingerprint,
            "input_fingerprint": getattr(self, "input_fingerprint", None),
            "completed_samples": sorted(r["sample_id"] for r in results),
            "consumed_budget": self.consumed_budget,
            "stage_versions":
                (self.snap.load_manifest() or {}).get("stage_versions", {}),
        })

    def _refuse_resume(self, message):
        record = {
            "sample_id": "<run>",
            "stage": "load_validate",
            "code": E_CONFIG_MISMATCH,
            "message": message,
            "injection_id": None,
        }
        self.failures = [record]
        self.pending = []
        self.completion_order = []
        self._write_outputs()
        return EXIT_CONFIG_MISMATCH

    def _write_outputs(self):
        out = self.output_dir
        results = sorted(self.completed.get("results", []),
                         key=lambda r: r["sample_id"])
        failures = _sort_failures(self.failures)
        ledger = self.completed.get("ledger", [])
        _write_jsonl(os.path.join(out, "results.jsonl"), results)
        _write_jsonl(os.path.join(out, "failures.jsonl"), failures)
        _write_jsonl(os.path.join(out, "budget_ledger.jsonl"), ledger)
        _write_jsonl(os.path.join(out, "cache_evictions.jsonl"),
                     self.cache.evictions)
        _write_jsonl(os.path.join(out, "cache_recomputes.jsonl"),
                     self.cache.recomputes)
        _write_json(os.path.join(out, "pending_queue.json"),
                    {"pending_samples": sorted(getattr(self, "pending", []))})
        ranking = sorted(results, key=lambda r: (-r["score"], r["sample_id"]))
        _write_json(os.path.join(out, "summary.json"), {
            "n_results": len(results),
            "n_failures": len(failures),
            "n_pending": len(getattr(self, "pending", [])),
            "consumed_budget": self.consumed_budget,
            "budget": self.budget,
            "decisions": {
                d: sum(1 for r in results if r["decision"] == d)
                for d in ("accept", "reject")
            },
            "borderline_samples": [r["sample_id"] for r in results
                                   if r["borderline"]],
            "ranking": [{"sample_id": r["sample_id"], "score": r["score"]}
                        for r in ranking],
        })
        _write_json(os.path.join(out, "diagnostics.json"), {
            "workers": self.workers,
            "budget": self.budget,
            "budget_source": self.budget_source,
            "config_fingerprint": self.cfg.fingerprint,
            "input_fingerprint": getattr(self, "input_fingerprint", None),
            "resumed": getattr(self, "resumed", False),
            "injections": self.plan.report(),
            "completion_order": getattr(self, "completion_order", []),
        })

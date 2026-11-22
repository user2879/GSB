"""端到端主管线：加载 -> 定点计算 -> 审计账本 -> 预算调度 -> 快照 -> 报告。

增量语义：以 (request_id, item_id) 的输入指纹命中缓存即复用，未命中重算；
最终产物只由确定性输入决定，因此增量运行与空白全量重建逐字段一致（要求 7）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .audit import build_request_ledger, highest_frac_reference
from .config import FusionConfig
from .engine import compute_request
from .loader import LoadedStream
from .scheduler import RecomputeScheduler, WorkUnit
from .serialization import content_hash
from .snapshot import (STAGE_BOUNDARY, STAGE_FLIPS, STAGE_LEDGER,
                       STAGE_REPORT, SnapshotStore)
from .fixedpoint import STAGE_ORDER


def _feature_lookup(stream: LoadedStream) -> Dict[str, List[float]]:
    lookup: Dict[str, List[float]] = {}
    dim_mismatch: Dict[str, str] = {}
    for rec in stream.records:
        if rec.item_id in lookup and lookup[rec.item_id] != rec.features:
            dim_mismatch[rec.item_id] = "同一物品跨通道特征不一致，取首次出现值"
            continue
        lookup.setdefault(rec.item_id, rec.features)
    return lookup


def request_fingerprint(request_id: str, stream: LoadedStream,
                        cfg: FusionConfig) -> str:
    payload = {
        "request_id": request_id,
        "items": stream.request_items.get(request_id, []),
        "channels": {
            chan: [
                {"item_id": r.item_id, "raw_score": r.raw_score,
                 "features": r.features, "channel_rank": r.channel_rank,
                 "shard": r.shard}
                for r in stream.requests[request_id].get(chan, [])
            ]
            for chan in sorted(stream.requests.get(request_id, {}))
        },
        "config": cfg.raw,
    }
    return content_hash(payload)


class Pipeline:
    def __init__(self, cfg: FusionConfig, store: SnapshotStore):
        self.cfg = cfg
        self.store = store
        # 缓存：request_id -> (fingerprint, engine_result, ledger)
        self._cache: Dict[str, Tuple[str, Dict[str, Any], Dict[str, Any]]] = {}

    def _compute_request(self, stream: LoadedStream, request_id: str):
        fp = request_fingerprint(request_id, stream, self.cfg)
        cached = self._cache.get(request_id)
        if cached and cached[0] == fp:
            return cached[1], cached[2], "cached"
        features = _feature_lookup(stream)
        result = compute_request(
            request_id, stream.requests[request_id],
            stream.request_items[request_id], features, self.cfg)
        ledger = build_request_ledger(result, max(self.cfg.frac_bits_choices))
        self._cache[request_id] = (fp, result, ledger)
        return result, ledger, "fresh"

    def run(self, stream: LoadedStream, batch_budget: int,
            start_version: int = 1) -> Dict[str, Any]:
        cfg = self.cfg
        request_ids = stream.request_ids
        units: List[WorkUnit] = []
        for req_id in request_ids:
            for item_id in stream.request_items[req_id]:
                units.append(WorkUnit(req_id, item_id))

        # 待办优先：上一批保留的单元仍在 scheduler.todo 队首。
        scheduler = RecomputeScheduler(
            units, cfg.concurrency,
            lambda u: self._compute_request(stream, u.request_id))

        batch_ledgers = []
        manifests = []
        version = start_version
        batches_since_commit = 0

        # 阶段化提交：每达到 snapshot_commit_interval_batches 批，
        # 就把当前已完成请求的产物整体替换提交一次。
        while not scheduler.drained:
            batch_ledgers.append(scheduler.run_batch(batch_budget).to_dict())
            batches_since_commit += 1
            drained = scheduler.drained
            if (batches_since_commit
                    >= cfg.snapshot_commit_interval_batches) or drained:
                completed = self._completed_requests(stream)
                request_results, request_ledgers = self._collect(
                    stream, completed)
                report = self._report(
                    request_results, request_ledgers, batch_ledgers,
                    highest_frac_reference(request_results, cfg), stream)
                manifest = self._commit(
                    version, cfg, completed, batch_ledgers,
                    self._ledger_payload(request_ledgers),
                    self._flips_payload(request_ledgers),
                    self._boundary_payload(request_ledgers), report)
                manifests.append(manifest)
                version += 1
                batches_since_commit = 0

        request_results, request_ledgers = self._collect(
            stream, request_ids)
        ledger_payload = self._ledger_payload(request_ledgers)
        flips_payload = self._flips_payload(request_ledgers)
        boundary_payload = self._boundary_payload(request_ledgers)
        report = self._report(
            request_results, request_ledgers, batch_ledgers,
            highest_frac_reference(request_results, cfg), stream)
        report["manifest"] = manifest
        report["intermediate_manifests"] = [
            m["global_version"] for m in manifests[:-1]]
        report["batch_ledgers"] = batch_ledgers
        return report

    def _completed_requests(self, stream: LoadedStream) -> List[str]:
        done = set()
        for req_id in stream.request_ids:
            if req_id in self._cache:
                done.add(req_id)
        return sorted(done)

    def _collect(self, stream: LoadedStream, request_ids: List[str]):
        request_results: Dict[str, Any] = {}
        request_ledgers: Dict[str, Any] = {}
        for req_id in request_ids:
            result, ledger, _ = self._compute_request(stream, req_id)
            request_results[req_id] = result
            request_ledgers[req_id] = ledger
        return request_results, request_ledgers

    # ---------- 产物组装 ----------
    def _ledger_payload(self, request_ledgers: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "requests": [request_ledgers[r] for r in sorted(request_ledgers)]
        }

    def _flips_payload(self, request_ledgers: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "requests": [
                {"request_id": r, "flips": request_ledgers[r]["flips"]}
                for r in sorted(request_ledgers)
            ]
        }

    def _boundary_payload(self, request_ledgers: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "requests": [
                {"request_id": r,
                 "boundary_pairs": request_ledgers[r]["boundary_pairs"]}
                for r in sorted(request_ledgers)
            ]
        }

    def _stage_max_diffs(self, request_results: Dict[str, Any]) -> Dict[str, Any]:
        # 统一到最高小数位宽的公共定点单位后再比较，避免尺度差异污染差值。
        max_f = max(self.cfg.frac_bits_choices)
        max_diff = {stage: 0 for stage in STAGE_ORDER}

        def at_unit(entry, stage):
            shift = max_f - entry["frac_bits"]
            value = entry["stages"][stage]
            if stage == "normalize":
                return {k: int(v) << shift for k, v in value.items()}
            return int(value) << shift

        for result in request_results.values():
            settings = result["settings"]
            for i in range(len(settings)):
                for j in range(i + 1, len(settings)):
                    ea = {e["item_id"]: e for e in settings[i]["entries"]}
                    eb = {e["item_id"]: e for e in settings[j]["entries"]}
                    for item_id in ea:
                        for stage in STAGE_ORDER:
                            va, vb = at_unit(ea[item_id], stage), \
                                at_unit(eb[item_id], stage)
                            if isinstance(va, dict):
                                diff = max(abs(vb[k] - va[k]) for k in va)
                            else:
                                diff = abs(vb - va)
                            max_diff[stage] = max(max_diff[stage], diff)
        return max_diff

    def _report(self, request_results, request_ledgers, batch_ledgers,
                reference, stream: LoadedStream) -> Dict[str, Any]:
        all_flips = [f for l in request_ledgers.values() for f in l["flips"]]
        flip_count = len(all_flips)
        same_frac_flips = sum(
            1 for f in all_flips
            if f["setting_before"]["frac_bits"]
            == f["setting_after"]["frac_bits"])
        boundary_count = sum(
            len(l["boundary_pairs"]) for l in request_ledgers.values())
        pending_count = sum(
            len(b["pending"]) for b in batch_ledgers)
        consumed = sum(b["consumed_points"] for b in batch_ledgers)
        return {
            "config_hash": cfg_hash(self.cfg),
            "request_count": len(request_results),
            "candidate_count": sum(
                len(items) for items in stream.request_items.values()),
            "stage_max_fixed_point_diff": self._stage_max_diffs(
                request_results),
            "flipped_candidate_count": flip_count,
            "flipped_same_frac_gear_count": same_frac_flips,
            "flipped_cross_frac_count": flip_count - same_frac_flips,
            "boundary_pair_count": boundary_count,
            "budget_consumed_points": consumed,
            "budget_pending_count": pending_count,
            "load_failures": [f.to_dict() for f in stream.failures],
            "highest_frac_reference": reference,
        }

    def _commit(self, version, cfg, request_ids, batch_ledgers,
                ledger_payload, flips_payload, boundary_payload, report):
        version_dir = self.store.new_version_dir(version)
        stage_versions: Dict[str, str] = {}
        stage_versions[STAGE_LEDGER] = self.store.write_stage(
            version_dir, STAGE_LEDGER, ledger_payload)
        stage_versions[STAGE_FLIPS] = self.store.write_stage(
            version_dir, STAGE_FLIPS, flips_payload)
        stage_versions[STAGE_BOUNDARY] = self.store.write_stage(
            version_dir, STAGE_BOUNDARY, boundary_payload)
        stage_versions[STAGE_REPORT] = self.store.write_stage(
            version_dir, STAGE_REPORT, report)
        manifest = {
            "global_version": version,
            "config_hash": cfg_hash(cfg),
            "completed_requests": sorted(request_ids),
            "consumed_recompute_points": sum(
                b["consumed_points"] for b in batch_ledgers),
            "stage_versions": stage_versions,
        }
        self.store.commit(version_dir, manifest)
        return manifest


def cfg_hash(cfg: FusionConfig) -> str:
    return cfg.hash()

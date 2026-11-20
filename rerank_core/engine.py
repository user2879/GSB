"""交互精排引擎：增量失效、预算响应、并发一致性、快照与报告。"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import cachekeys
from .clock import MonotonicClock
from .commands import Command, apply_command_to_config
from .config import RerankConfig, groups_fingerprint
from .errors import ConfigInvalidError, RecordInvalidError
from .fixedpoint import stable_hash
from .loader import load_candidates
from .models import Candidate, Failure
from .pipeline import StageArtifacts, compute_request_stages
from .serialize import (
    discount_to_dict,
    fusion_to_dict,
    rerank_to_dict,
)
from .snapshot import (
    SnapshotStore,
    STAGE_FUSION,
    STAGE_NORMALIZE,
    STAGE_RERANK,
)

ALL_STAGES = (STAGE_NORMALIZE, STAGE_FUSION, STAGE_RERANK)


@dataclass
class PendingWork:
    command_seq: int
    affected: list[str]
    stages_by_request: dict[str, list[str]]
    completed_requests: list[str] = field(default_factory=list)
    delivered: bool = False


@dataclass
class InteractionResponse:
    seq: int
    code: str
    completed_requests: int
    total_requests: int
    top_item: str | None
    affected_requests: list[str]
    progress: float
    reused: list[dict]
    recomputed: list[dict]
    message: str
    pending: bool
    budget_ms: float
    elapsed_ms: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class RerankEngine:
    def __init__(
        self,
        config: RerankConfig,
        candidates: dict[str, list[Candidate]],
        failures: list[Failure],
        snapshot_store: SnapshotStore,
        clock: MonotonicClock | None = None,
        *,
        resume: bool = False,
    ):
        self.config = config
        self.candidates = {rid: list(items) for rid, items in candidates.items()}
        self.failures = sorted(
            [dataclasses.replace(f) for f in failures],
            key=lambda f: (f.scope, f.target, f.line_no or 0, f.message),
        )
        self.store = snapshot_store
        self.clock = clock or MonotonicClock()
        self.request_order = list(self.candidates.keys())
        self.global_version = 0
        # 内存态缓存：缓存键 -> 产物；与快照内容寻址键一致
        self.cache: dict[str, dict] = {}
        self.request_stage_keys: dict[str, dict[str, object]] = {}
        self.latest_results: dict[str, StageArtifacts] = {}
        self.pending: PendingWork | None = None
        self.command_log: list[dict] = []
        self._commands_since_commit = 0
        self.stage_timings: dict[str, float] = {stage: 0.0 for stage in ALL_STAGES}
        self.injection_hits: list[dict] = []
        if resume and self.store.exists():
            self._restore()

    # ------------------------------------------------------------------
    # 快照恢复
    # ------------------------------------------------------------------
    def _restore(self) -> None:
        manifest = self.store.verify_or_reject(self.config, self.candidates_fingerprint())
        self.global_version = manifest["global_version"]
        self.request_stage_keys = manifest.get("stage_cache_keys", {})
        from .pipeline import StageArtifacts
        from .serialize import fusion_from_dict, rerank_from_dict

        for request_id, key_map in self.request_stage_keys.items():
            normalize_payload = self.store.read_artifact(key_map["normalize"])
            self.cache[key_map["normalize"]] = normalize_payload
            fusion_payloads: dict[str, dict] = {}
            for item_id, key in key_map["fusion"].items():
                payload = self.store.read_artifact(key)
                self.cache[key] = payload
                fusion_payloads[item_id] = payload
            rerank_payload = self.store.read_artifact(key_map["rerank"])
            self.cache[key_map["rerank"]] = rerank_payload

            fusion_results = [
                fusion_from_dict(fusion_payloads[c.item_id])
                for c in self.candidates[request_id]
                if c.item_id in fusion_payloads
            ]
            from .fusion import sort_fusion

            fusion_results = sort_fusion(
                fusion_results, self.candidates[request_id], self.config
            )
            rerank_result = rerank_from_dict(rerank_payload)
            self.latest_results[request_id] = StageArtifacts(
                request_id=request_id,
                request_fp=cachekeys.request_content_fingerprint(
                    request_id, self.candidates[request_id], self.config
                ),
                normalize_key=key_map["normalize"],
                fusion_keys=dict(key_map["fusion"]),
                rerank_key=key_map["rerank"],
                discount_keys=list(rerank_payload.get("discount_keys", [])),
                stats=normalize_payload["stats"],
                quantized=normalize_payload["quantized"],
                fusion_results=fusion_results,
                fusion_order=[r.item_id for r in fusion_results],
                rerank=rerank_result,
            )

    def candidates_fingerprint(self) -> str:
        payload = {
            rid: [
                [
                    c.item_id,
                    [c.recall_scores[ch] for ch in self.config.recall_channels],
                    list(c.dense),
                    list(c.sparse),
                ]
                for c in sorted(items, key=lambda c: c.item_id)
            ]
            for rid, items in sorted(self.candidates.items())
        }
        return stable_hash(["candidates", payload])

    # ------------------------------------------------------------------
    # 命令应用与失效范围判定
    # ------------------------------------------------------------------
    def apply_command(self, command: Command) -> list[str]:
        """应用命令，更新配置/候选集，返回受影响请求 ID 列表（请求首次出现顺序）。"""
        affected: list[str] = []
        if command.type in ("add_candidate", "delete_candidate"):
            affected = self._apply_candidate_command(command)
        else:
            if command.type != "poll":
                self.config = apply_command_to_config(self.config, command)
            if command.type == "set_weight":
                # 仅融合与下游失效；归一化统计量可复用
                affected = list(self.request_order)
                self._invalidate(affected, (STAGE_FUSION, STAGE_RERANK))
            elif command.type in ("set_normalization", "set_quant"):
                affected = list(self.request_order)
                self._invalidate(affected, ALL_STAGES)
            elif command.type in ("set_rerank_window", "set_penalty"):
                affected = list(self.request_order)
                self._invalidate(affected, (STAGE_RERANK,))
            elif command.type == "poll":
                affected = list(self.pending.affected) if self.pending else []
        return affected

    def _invalidate(self, requests: list[str], stages: tuple[str, ...]) -> None:
        for request_id in requests:
            key_map = self.request_stage_keys.get(request_id)
            if not key_map:
                continue
            for stage in stages:
                if stage == STAGE_NORMALIZE:
                    self.cache.pop(key_map["normalize"], None)
                elif stage == STAGE_FUSION:
                    for key in key_map["fusion"].values():
                        self.cache.pop(key, None)
                elif stage == STAGE_RERANK:
                    self.cache.pop(key_map["rerank"], None)
            self.latest_results.pop(request_id, None)

    def _apply_candidate_command(self, command: Command) -> list[str]:
        payload = command.payload
        request_id = payload["request_id"]
        if command.type == "add_candidate":
            raw = payload["record"]
            from .loader import _validate_record

            try:
                candidate = _validate_record(raw, payload.get("line_no", -1), self.config)
            except RecordInvalidError as exc:
                self.failures.append(
                    Failure(exc.code, "request", request_id, exc.message,
                            payload.get("line_no"))
                )
                self.failures.sort(
                    key=lambda f: (f.scope, f.target, f.line_no or 0, f.message)
                )
                self.injection_hits.append(
                    {"command_seq": command.seq, "code": exc.code,
                     "target": request_id, "message": exc.message}
                )
                return []
            bucket = self.candidates.setdefault(request_id, [])
            if any(c.item_id == candidate.item_id for c in bucket):
                failure = Failure(
                    "E_RECORD_INVALID", "request", request_id,
                    f"候选物品 ID 在请求内重复: {candidate.item_id}",
                    candidate.line_no,
                )
                self.failures.append(failure)
                self.failures.sort(
                    key=lambda f: (f.scope, f.target, f.line_no or 0, f.message)
                )
                self.injection_hits.append(
                    {"command_seq": command.seq, "code": failure.code,
                     "target": request_id, "message": failure.message}
                )
                return []
            bucket.append(candidate)
            bucket.sort(key=lambda c: c.item_id)
            if request_id not in self.request_order:
                self.request_order.append(request_id)
        else:
            item_id = payload["item_id"]
            bucket = self.candidates.get(request_id, [])
            before = len(bucket)
            self.candidates[request_id] = [c for c in bucket if c.item_id != item_id]
            if len(self.candidates[request_id]) == before:
                return []  # 未命中任何候选，不产生失效
            if not self.candidates[request_id]:
                self.candidates.pop(request_id)
                self.request_order.remove(request_id)
        # 候选变化只影响该请求的全部阶段，其他请求原样复用
        self._invalidate([request_id], ALL_STAGES)
        return [request_id]

    # ------------------------------------------------------------------
    # 阶段执行（并发 + 预算）
    # ------------------------------------------------------------------
    def _plan_request(self, request_id: str) -> tuple[list[str], dict]:
        """根据缓存键命中情况决定该请求哪些阶段需要重算。"""
        candidates = self.candidates[request_id]
        request_fp = cachekeys.request_content_fingerprint(
            request_id, candidates, self.config
        )
        normalize_key = cachekeys.normalize_cache_key(
            request_fp, cachekeys.normalization_param_fingerprint(self.config)
        )
        fusion_param_fp = cachekeys.fusion_param_fingerprint(self.config)
        fusion_keys = {
            candidate.item_id: cachekeys.fusion_cache_key(
                request_fp,
                cachekeys.item_feature_fingerprint(candidate, self.config),
                candidate.item_id,
                fusion_param_fp,
            )
            for candidate in candidates
        }
        previous = self.request_stage_keys.get(request_id, {})
        stages: list[str] = []

        if normalize_key not in self.cache:
            stages = [STAGE_NORMALIZE, STAGE_FUSION, STAGE_RERANK]
        else:
            fusion_changed = (
                previous.get("fusion") != fusion_keys
                or any(key not in self.cache for key in fusion_keys.values())
            )
            if fusion_changed:
                stages = [STAGE_FUSION, STAGE_RERANK]
            else:
                # 融合产物未变，依据当前融合顺序与重排参数派生重排键探测
                order = self._current_fusion_order_probe(request_id, fusion_keys)
                rerank_key = cachekeys.rerank_cache_key(
                    request_fp,
                    cachekeys.stable_hash(order),
                    cachekeys.rerank_param_fingerprint(self.config),
                )
                if (
                    previous.get("rerank") != rerank_key
                    or rerank_key not in self.cache
                ):
                    stages = [STAGE_RERANK]
        return stages, {"normalize": normalize_key, "fusion": fusion_keys}

    def _current_fusion_order_probe(self, request_id, fusion_keys):
        """从缓存融合产物还原融合排序物品顺序，用于派生重排缓存键。"""
        from .serialize import fusion_from_dict
        from .fusion import sort_fusion

        candidates = self.candidates[request_id]
        results = [
            fusion_from_dict(self.cache[fusion_keys[c.item_id]])
            for c in candidates
            if fusion_keys[c.item_id] in self.cache
        ]
        if len(results) != len(candidates):
            return []
        ordered = sort_fusion(results, candidates, self.config)
        return [r.item_id for r in ordered]

    def _execute_request(
        self, request_id: str, planned_stages: list[str], planned_keys: dict
    ) -> StageArtifacts:
        candidates = self.candidates[request_id]
        need_normalize = STAGE_NORMALIZE in planned_stages
        need_fusion = STAGE_FUSION in planned_stages
        need_rerank = STAGE_RERANK in planned_stages
        stage_wall: dict[str, int] = {}
        stage_begin = self.clock.now_ns()

        cached_normalize = None if need_normalize else self.cache.get(
            planned_keys["normalize"]
        )
        cached_fusion = None
        if not need_fusion:
            cached_fusion = {
                item_id: self.cache[key]
                for item_id, key in planned_keys["fusion"].items()
            }
        cached_rerank = None
        if not need_rerank:
            rerank_key = self.request_stage_keys.get(request_id, {}).get("rerank")
            cached_rerank = self.cache.get(rerank_key) if rerank_key else None

        mark = self.clock.now_ns()
        artifacts = compute_request_stages(
            request_id,
            candidates,
            self.config,
            need_normalize=need_normalize,
            cached_normalize=cached_normalize,
            need_fusion=need_fusion,
            cached_fusion=cached_fusion,
            need_rerank=need_rerank,
        )
        stage_wall["compute"] = self.clock.now_ns() - mark
        if cached_rerank is not None and not need_rerank:
            from .serialize import rerank_from_dict

            artifacts.rerank = rerank_from_dict(cached_rerank)
            artifacts.rerank_key = self.request_stage_keys[request_id]["rerank"]
            artifacts.discount_keys = [
                cachekeys.discount_cache_key(
                    artifacts.rerank_key, d.step, d.target_item, d.reference_item
                )
                for d in artifacts.rerank.discounts
            ]
        if planned_stages:
            share_ns = stage_wall["compute"] // len(planned_stages)
            for stage in planned_stages:
                self.stage_timings[stage] += share_ns / 1_000_000.0
        self._store_artifacts(artifacts, planned_stages)
        return artifacts

    def _store_artifacts(self, artifacts: StageArtifacts, planned_stages: list[str]) -> None:
        staged: list[tuple[Path, str]] = []
        if STAGE_NORMALIZE in planned_stages:
            payload = {
                "key": artifacts.normalize_key,
                "stats": [s.to_dict() for s in artifacts.stats],
                "quantized": artifacts.quantized,
                "item_order": [c.item_id for c in self.candidates[artifacts.request_id]],
            }
            self.cache[artifacts.normalize_key] = payload
            staged.append((self.store.stage_artifact(artifacts.normalize_key, payload),
                           artifacts.normalize_key))
        if STAGE_FUSION in planned_stages:
            fusion_payload = {}
            for result in artifacts.fusion_results:
                payload = fusion_to_dict(result)
                key = artifacts.fusion_keys[result.item_id]
                payload["key"] = key
                fusion_payload[result.item_id] = payload
                self.cache[key] = payload
                staged.append((self.store.stage_artifact(key, payload), key))
        if STAGE_RERANK in planned_stages:
            payload = rerank_to_dict(artifacts.rerank)
            payload["key"] = artifacts.rerank_key
            payload["discount_keys"] = artifacts.discount_keys
            payload["discounts"] = [
                discount_to_dict(d) for d in artifacts.rerank.discounts
            ]
            self.cache[artifacts.rerank_key] = payload
            staged.append((self.store.stage_artifact(artifacts.rerank_key, payload),
                           artifacts.rerank_key))
        self._pending_staged = getattr(self, "_pending_staged", []) + staged

        previous = self.request_stage_keys.get(artifacts.request_id, {})
        self.request_stage_keys[artifacts.request_id] = {
            "normalize": artifacts.normalize_key,
            "fusion": dict(artifacts.fusion_keys),
            "rerank": artifacts.rerank_key,
        }
        self.latest_results[artifacts.request_id] = artifacts

    def run_with_budget(
        self,
        command: Command,
        affected: list[str],
        deadline_ns: int,
        start_ns: int | None = None,
    ) -> InteractionResponse:
        start_ns = self.clock.now_ns() if start_ns is None else start_ns
        reused: list[dict] = []
        recomputed: list[dict] = []
        plans: dict[str, list[str]] = {}
        planned_keys: dict[str, dict] = {}

        # 计划阶段本身只读，顺序确定，先产出复用/重算清单
        for request_id in self.request_order:
            stages, keys = self._plan_request(request_id)
            plans[request_id] = stages
            planned_keys[request_id] = keys
            if stages:
                recomputed.append({"request_id": request_id, "stages": list(stages)})
            else:
                reused.append({"request_id": request_id,
                               "stages": list(ALL_STAGES)})

        pending_targets = [rid for rid in self.request_order if plans[rid]]
        completed: list[str] = []
        budget_exceeded = False
        self._pending_staged = []

        def worker(request_id: str) -> StageArtifacts:
            return self._execute_request(
                request_id, plans[request_id], planned_keys[request_id]
            )

        if pending_targets:
            max_workers = min(self.config.concurrency, len(pending_targets))
            pool = ThreadPoolExecutor(max_workers=max_workers)
            try:
                # 有界在途窗口：在途任务数绝不超过配置并发数
                in_flight: dict = {}
                cursor = 0

                def submit_next() -> None:
                    nonlocal cursor
                    while len(in_flight) < max_workers and cursor < len(pending_targets):
                        request_id = pending_targets[cursor]
                        in_flight[pool.submit(worker, request_id)] = request_id
                        cursor += 1

                submit_next()
                while in_flight:
                    if self.clock.now_ns() >= deadline_ns:
                        # 已在途任务允许跑完；仅当仍有未提交任务时才算挂起
                        future = next(iter(in_flight))
                        if not future.done():
                            budget_exceeded = True
                            break
                    future = next(iter(in_flight))
                    try:
                        future.result(timeout=0.002)
                    except TimeoutError:
                        continue
                    request_id = in_flight.pop(future)
                    completed.append(request_id)
                    submit_next()
            finally:
                if budget_exceeded:
                    for future in in_flight:
                        future.cancel()
                pool.shutdown(wait=True)
            completed.sort(key=lambda rid: self.request_order.index(rid))

        finished_all = not budget_exceeded and len(completed) == len(pending_targets)
        elapsed_ms = (self.clock.now_ns() - start_ns) / 1_000_000

        if not finished_all:
            unfinished = [rid for rid in pending_targets if rid not in completed]
            self.pending = PendingWork(
                command_seq=command.seq,
                affected=affected,
                stages_by_request={rid: plans[rid] for rid in unfinished},
                completed_requests=completed,
            )
            response = self._build_response(
                command, affected, completed, reused, recomputed,
                pending=True, elapsed_ms=elapsed_ms,
            )
            return response

        self.pending = None
        self._commands_since_commit += 1
        if self._commands_since_commit >= self.config.snapshot_every_commands:
            self._commit_snapshot(command)
            self._commands_since_commit = 0
        return self._build_response(
            command, affected, self.request_order, reused, recomputed,
            pending=False, elapsed_ms=elapsed_ms,
        )

    def _reconcile_staged_artifacts(self) -> None:
        """保证所有当前生效的缓存键在提交前都有磁盘产物。

        预算挂起期间已完成但未提交的产物仅存在于内存缓存与 tmp，
        提交时缓存命中不会重新计算，这里按 request_stage_keys 统一补齐。
        """
        staged = getattr(self, "_pending_staged", [])
        staged_keys = {key for _, key in staged}
        for request_id in self.request_order:
            key_map = self.request_stage_keys.get(request_id)
            if not key_map:
                continue
            all_keys = [key_map["normalize"], key_map["rerank"]]
            all_keys += list(key_map["fusion"].values())
            for key in all_keys:
                if key in staged_keys or self.store.has_artifact(key):
                    continue
                payload = self.cache.get(key)
                if payload is None:
                    continue
                staged.append((self.store.stage_artifact(key, payload), key))
                staged_keys.add(key)
        self._pending_staged = staged

    # ------------------------------------------------------------------
    # 响应、快照提交与补交
    # ------------------------------------------------------------------
    def _top_item(self) -> str | None:
        if not self.request_order:
            return None
        first_request = self.request_order[0]
        artifacts = self.latest_results.get(first_request)
        if artifacts is None or not artifacts.rerank.final_order:
            return None
        return artifacts.rerank.final_order[0]

    def _build_response(
        self, command, affected, completed_requests, reused, recomputed,
        *, pending, elapsed_ms,
    ) -> InteractionResponse:
        total = len(self.request_order)
        completed_count = len(completed_requests)
        code = "E_INTERACT_BUDGET_PENDING" if pending else "OK"
        message = (
            "预算内未完成全部重算，返回最近完成结果，后续命令补交"
            if pending else "所有受影响阶段已按最新参数完成"
        )
        return InteractionResponse(
            seq=command.seq,
            code=code,
            completed_requests=completed_count,
            total_requests=total,
            top_item=self._top_item(),
            affected_requests=list(affected),
            progress=round(completed_count / total, 6) if total else 1.0,
            reused=reused,
            recomputed=recomputed,
            message=message,
            pending=pending,
            budget_ms=self.config.latency_budget_ms,
            elapsed_ms=round(elapsed_ms, 6),
        )

    def _commit_snapshot(self, command: Command) -> None:
        self._reconcile_staged_artifacts()
        self.global_version += 1
        stage_versions = {
            stage: stable_hash(
                [
                    self.request_stage_keys.get(rid, {}).get(stage)
                    for rid in self.request_order
                ]
            )
            for stage in ALL_STAGES
        }
        self.store.commit(
            global_version=self.global_version,
            config=self.config,
            candidates_fingerprint=self.candidates_fingerprint(),
            completed_cursor=list(self.request_order),
            stage_versions=stage_versions,
            staged_files=getattr(self, "_pending_staged", []),
            stage_cache_keys={rid: dict(keys)
                              for rid, keys in self.request_stage_keys.items()},
            extra={
                "last_command_seq": command.seq,
                "stage_timings_ms": dict(self.stage_timings),
            },
        )
        self._pending_staged = []

    def handle_command(self, command: Command) -> InteractionResponse:
        """交互入口：先补交挂起工作，再应用新命令，并在单调钟预算内返回。"""
        start_ns = self.clock.now_ns()
        budget_ns = int(self.config.latency_budget_ms * 1_000_000)
        deadline_ns = start_ns + budget_ns

        # 上一条命令挂起的重算优先在本次预算内补交
        if self.pending is not None and command.type != "poll":
            # 新命令改变状态时，旧挂起范围由新计划重新判定
            self.pending = None

        affected = self.apply_command(command)
        response = self.run_with_budget(command, affected, deadline_ns, start_ns)
        self.command_log.append(response.to_dict())
        return response

    def drain_pending(self, command: Command) -> InteractionResponse:
        """``poll`` 命令：在新预算内继续完成挂起重算并补交结果。"""
        start_ns = self.clock.now_ns()
        deadline_ns = start_ns + int(self.config.latency_budget_ms * 1_000_000)
        affected = self.apply_command(command)
        response = self.run_with_budget(command, affected, deadline_ns, start_ns)
        self.command_log.append(response.to_dict())
        return response

    # ------------------------------------------------------------------
    # 诊断报告
    # ------------------------------------------------------------------
    def flush_snapshot(self) -> None:
        """显式把未提交的阶段产物整体落盘（不改变业务状态）。"""
        if self._commands_since_commit > 0 or not self.store.exists():
            self._commit_snapshot(Command(seq=-1, type="poll", payload={}))
            self._commands_since_commit = 0

    def build_report(self, *, high_tier: bool = True) -> dict:
        if self._commands_since_commit > 0:
            self._commit_snapshot(Command(seq=-1, type="poll", payload={}))
            self._commands_since_commit = 0
        per_request: list[dict] = []
        for request_id in self.request_order:
            artifacts = self.latest_results[request_id]
            high_diff = self._high_tier_diff(request_id) if high_tier else {}
            per_request.append({
                "request_id": request_id,
                "request_content_fp": artifacts.request_fp,
                "normalize_cache_key": artifacts.normalize_key,
                "fusion_cache_keys": dict(artifacts.fusion_keys),
                "rerank_cache_key": artifacts.rerank_key,
                "discount_cache_keys": list(artifacts.discount_keys),
                "fusion_order": list(artifacts.fusion_order),
                "fusion_scores": {
                    result.item_id: result.score_fixed
                    for result in artifacts.fusion_results
                },
                "contributions": {
                    result.item_id: [
                        {
                            "feature_index": c.feature_index,
                            "feature_name": c.feature_name,
                            "group": c.group,
                            "normalized_fixed": c.normalized_fixed,
                            "weight_fixed": c.weight_fixed,
                            "contribution_fixed": c.contribution_fixed,
                        }
                        for c in result.contributions
                    ]
                    for result in artifacts.fusion_results
                },
                "rerank": rerank_to_dict(artifacts.rerank),
                "high_tier_score_diff": high_diff,
            })
        return {
            "config_hash": self.config.fingerprint(),
            "groups_hash": groups_fingerprint(self.config),
            "candidates_hash": self.candidates_fingerprint(),
            "global_version": self.global_version,
            "frac_bits": self.config.frac_bits,
            "quant_levels": self.config.quant_levels,
            "request_order": list(self.request_order),
            "per_request": per_request,
            "stage_timings_ms": {k: round(v, 6) for k, v in self.stage_timings.items()},
            "failures": [f.to_dict() for f in self.failures],
            "injection_hits": list(self.injection_hits),
            "interactions": list(self.command_log),
        }

    def _high_tier_diff(self, request_id: str) -> dict[str, int]:
        """按高一档小数位宽重算融合分，返回与当前档位的整数差值。"""
        candidates = self.candidates[request_id]
        high_bits = min(self.config.frac_bits + 1, 12)
        from .normalize import compute_stats, quantize_request
        from .fusion import fuse_request, sort_fusion

        from .normalize import raw_matrix

        matrix_stats = compute_stats(
            request_id,
            raw_matrix(candidates),
            self.config.feature_dim,
            self.config.normalization,
        )
        high_quantized = quantize_request(
            request_id, candidates, matrix_stats,
            dataclasses.replace(self.config, frac_bits=high_bits),
        )
        high_results = fuse_request(
            candidates, high_quantized, self.config,
            frac_bits=high_bits, quant_levels=self.config.quant_levels,
        )
        current = {r.item_id: r.score_fixed for r in
                   self.latest_results[request_id].fusion_results}
        scale_ratio = 10 ** (high_bits - self.config.frac_bits)
        return {
            result.item_id: result.score_fixed - current[result.item_id] * scale_ratio
            for result in high_results
        }

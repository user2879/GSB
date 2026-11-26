from __future__ import annotations

import heapq
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .canonical import fingerprint
from .client import NodeProcessManager
from .errors import BizError, make_error
from .fixedpoint import ROUNDING_POLICY, evaluate_watermark
from .generator import samples_fingerprint
from .keys import (
    aggregate_key,
    readiness_key,
    receipt_key,
    watermark_key,
)
from .models import ProjectConfig, ShardSpec
from .persistence import SnapshotStore, termination_classification
from .state import MaterializedState


class ForcedTermination(Exception):
    def __init__(self, completed: int):
        super().__init__(f"forced termination after {completed} shards")
        self.completed = completed


@dataclass
class SchedulerCounters:
    completed_cursor: list[str] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    committed_units: int = 0
    dispatch_units: int = 0
    current_batch_index: int = 0
    current_batch_points: int = 0
    batch_plan: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_cursor": list(self.completed_cursor),
            "pending": list(self.pending),
            "committed_units": self.committed_units,
            "dispatch_units": self.dispatch_units,
            "current_batch_index": self.current_batch_index,
            "current_batch_points": self.current_batch_points,
            "batch_plan": list(self.batch_plan),
        }


def _shard_points(shard: ShardSpec) -> int:
    return sum(batch.sample_count * shard.dimensions for batch in shard.batches)


def _resource_units(shard: ShardSpec, points_per_unit: int) -> int:
    points = _shard_points(shard)
    return max(1, math.ceil(points / points_per_unit))


def _topological_order(state: MaterializedState) -> tuple[list[str], list[dict[str, Any]]]:
    shard_ids = set(state.shards)
    indegree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {shard_id: [] for shard_id in shard_ids}
    invalid: list[dict[str, Any]] = []
    for shard_id in sorted(shard_ids):
        shard = state.shards[shard_id]
        valid_upstream = []
        for dependency in shard.upstream:
            if dependency in shard_ids and dependency != shard_id:
                valid_upstream.append(dependency)
                dependents[dependency].append(shard_id)
            else:
                invalid.append(
                    make_error(
                        "E_DEP_MISSING",
                        "shard",
                        shard_id,
                        "upstream",
                        f"上游 {dependency} 不存在",
                    ).to_dict()
                )
        indegree[shard_id] = len(valid_upstream)
    ready = [shard_id for shard_id in sorted(shard_ids) if indegree[shard_id] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        shard_id = heapq.heappop(ready)
        order.append(shard_id)
        for dependent in sorted(dependents[shard_id]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    unprocessed = sorted(shard_ids - set(order))
    for shard_id in unprocessed:
        invalid.append(
            make_error(
                "E_DEP_CYCLE",
                "shard",
                shard_id,
                "upstream",
                "分片未能进入拓扑序，存在循环依赖",
            ).to_dict()
        )
    return order + unprocessed, invalid


def _content_resolution_plan(state: MaterializedState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_shards = {
        shard.shard_id: shard.to_dict() for shard in state.base_config.shards
    }
    initial_quota = next(
        (
            quota
            for quota in state.base_config.quota_versions
            if quota.version == state.base_config.active_quota_version
        ),
        None,
    )
    changed: set[str] = {
        shard_id
        for shard_id, revision in state.shard_revisions.items()
        if revision > 1
    }
    for shard_id, shard in state.shards.items():
        if shard_id not in initial_shards or initial_shards[shard_id] != shard.to_dict():
            changed.add(shard_id)
    impacted = set(changed)
    propagate = True
    while propagate:
        propagate = False
        for shard_id, shard in state.shards.items():
            if shard_id not in impacted and any(
                dependency in impacted for dependency in shard.upstream
            ):
                impacted.add(shard_id)
                propagate = True
    quota_changed = (
        initial_quota is None
        or state.active_quota.to_dict() != initial_quota.to_dict()
    )
    width_changed = initial_quota is not None and state.width != initial_quota.width
    node_changes: dict[str, list[str]] = {}
    for node_id in sorted(state.nodes):
        initial_node = next(
            node for node in state.base_config.nodes if node.node_id == node_id
        )
        initial_capacity = (
            initial_quota.capacity_slots.get(node_id, initial_node.capacity_slots)
            if initial_quota is not None
            else initial_node.capacity_slots
        )
        initial_lease = (
            initial_quota.lease_limits.get(node_id, initial_node.lease_limit)
            if initial_quota is not None
            else initial_node.lease_limit
        )
        effective = state.effective_node(node_id)
        reasons: list[str] = []
        if effective.capacity_slots != initial_capacity:
            reasons.append("NODE_CAPACITY_CHANGED")
        if effective.lease_limit != initial_lease:
            reasons.append("NODE_LEASE_LIMIT_CHANGED")
        if reasons:
            node_changes[node_id] = reasons
    reuse_plan: list[dict[str, Any]] = []
    invalidation_plan: list[dict[str, Any]] = []
    for shard_id in sorted(state.shards):
        shard = state.shards[shard_id]
        reasons: list[str] = []
        if shard_id in changed:
            reasons.append("SHARD_CONTENT_CHANGED")
        changed_upstream = sorted(
            dependency for dependency in shard.upstream if dependency in impacted
        )
        if changed_upstream:
            reasons.append("UPSTREAM_CHANGED:" + ",".join(changed_upstream))
        if quota_changed:
            reasons.append("QUOTA_VERSION_CHANGED")
        if width_changed:
            reasons.append("FIXEDPOINT_WIDTH_CHANGED")
        reasons.extend(node_changes.get(shard.node_id, []))
        if reasons:
            invalidation_plan.append(
                {
                    "shard_id": shard_id,
                    "decision": "INVALIDATE",
                    "reasons": reasons,
                    "downstream_affected": shard_id in impacted - changed,
                }
            )
        else:
            reuse_plan.append(
                {
                    "shard_id": shard_id,
                    "decision": "REUSE",
                    "reasons": ["CONTENT_AND_UPSTREAM_FINGERPRINTS_UNCHANGED"],
                }
            )
    return reuse_plan, invalidation_plan


def _stage_resolution_ledgers(
    reuse_plan: list[dict[str, Any]],
    invalidation_plan: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reuse: list[dict[str, Any]] = []
    invalidation: list[dict[str, Any]] = []
    stage_order = ("readiness", "watermark", "aggregate", "receipt")
    for entry in reuse_plan:
        for stage in stage_order:
            reuse.append(
                {
                    "stage": stage,
                    "shard_id": entry["shard_id"],
                    "decision": "REUSE",
                    "reasons": list(entry["reasons"]),
                }
            )
    for entry in invalidation_plan:
        reasons = list(entry["reasons"])
        for stage in stage_order:
            applicable = list(reasons)
            if stage == "readiness":
                applicable = [
                    reason
                    for reason in applicable
                    if not reason.startswith("FIXEDPOINT_WIDTH_CHANGED")
                    and reason != "SHARD_CONTENT_CHANGED"
                    and reason != "NODE_CAPACITY_CHANGED"
                    and reason != "NODE_LEASE_LIMIT_CHANGED"
                ]
            elif stage == "watermark":
                applicable = [
                    reason
                    for reason in applicable
                    if not reason.startswith("UPSTREAM_CHANGED:")
                ]
            if applicable:
                invalidation.append(
                    {
                        "stage": stage,
                        "shard_id": entry["shard_id"],
                        "decision": "INVALIDATE",
                        "reasons": applicable,
                    }
                )
            else:
                reuse.append(
                    {
                        "stage": stage,
                        "shard_id": entry["shard_id"],
                        "decision": "REUSE",
                        "reasons": ["STAGE_INPUTS_UNCHANGED"],
                    }
                )
    return reuse, invalidation


def _cache_records_for_shard(cache: dict[str, Any], stage: str, shard_id: str) -> list[dict[str, Any]]:
    stage_map = cache.get(stage, {})
    return [
        record
        for record in stage_map.values()
        if record.get("shard_id") == shard_id
    ]


def _first_invalidation_reason(
    previous_records: list[dict[str, Any]], current: dict[str, Any]
) -> dict[str, Any]:
    if not previous_records:
        return {
            "reason": "NO_PRIOR_ARTIFACT",
            "field": "cache_key",
            "previous": None,
            "current": current.get("cache_key"),
        }
    previous = sorted(previous_records, key=lambda item: str(item.get("cache_key")))[-1]
    fields = (
        "shard_id",
        "upstream",
        "quota_version",
        "quota_fingerprint",
        "width",
        "lease_generation",
    )
    for field_name in fields:
        if previous.get(field_name) != current.get(field_name):
            return {
                "reason": f"CONTENT_CHANGED:{field_name}",
                "field": field_name,
                "previous": previous.get(field_name),
                "current": current.get(field_name),
            }
    return {
        "reason": "CACHE_KEY_CHANGED",
        "field": "cache_key",
        "previous": previous.get("cache_key"),
        "current": current.get("cache_key"),
    }


def _execute_shard(
    *,
    state: MaterializedState,
    manager: NodeProcessManager,
    shard: ShardSpec,
    quota_fingerprint: str,
    width: int,
    lease_generation: int,
    upstream_summary_fingerprints: list[str],
) -> dict[str, Any]:
    node = state.effective_node(shard.node_id)
    samples = state.samples(shard.shard_id)
    execution: dict[str, Any] = {
        "node_id": node.node_id,
        "lease_generation": None,
        "node_lease_generation": None,
        "elapsed_monotonic_ms": None,
        "acquire": None,
        "release": None,
        "request_id": f"req-{uuid.uuid5(uuid.NAMESPACE_URL, state.state_fingerprint + shard.shard_id)}",
    }
    status = manager.request(node.node_id, {"operation": "status"})
    if not status.get("ok"):
        return {"ok": False, "error": status, "execution": execution}
    acquire = manager.request(
        node.node_id,
        {
            "operation": "acquire",
            "shard_id": shard.shard_id,
            "ttl_ms": state.base_config.lease_ttl_ms,
            "expected_generation": status.get("lease_generation"),
        },
    )
    execution["acquire"] = acquire
    if not acquire.get("ok"):
        return {"ok": False, "error": acquire, "execution": execution}
    lease_id = acquire["lease_id"]
    expected_generation = int(acquire["lease_generation"])
    execution["lease_id"] = lease_id
    execution["node_lease_generation"] = expected_generation
    aggregate_response: dict[str, Any] | None = None
    try:
        aggregate_response = manager.request(
            node.node_id,
            {
                "operation": "aggregate",
                "request_id": execution["request_id"],
                "shard_id": shard.shard_id,
                "batch_id": "joined:" + ",".join(
                    batch.batch_id for batch in sorted(shard.batches, key=lambda item: item.batch_id)
                ),
                "quota_version": state.active_quota_version,
                "width": width,
                "fixed_scale": 1000,
                "dimensions": shard.dimensions,
                "samples": samples,
                "lease_id": lease_id,
                "expected_generation": expected_generation,
                "upstream_summary_fingerprints": upstream_summary_fingerprints,
            },
        )
    finally:
        release = manager.request(
            node.node_id,
            {
                "operation": "release",
                "lease_id": lease_id,
                "expected_generation": expected_generation,
            },
        )
        execution["release"] = release
    if aggregate_response is None:
        return {
            "ok": False,
            "error": make_error(
                "E_INTERNAL", "shard", shard.shard_id, "aggregate", "节点未返回聚合结果"
            ).to_dict(),
            "execution": execution,
        }
    execution["elapsed_monotonic_ms"] = aggregate_response.get("elapsed_monotonic_ms")
    execution["node_lease_generation"] = aggregate_response.get("lease_generation")
    execution["duplicate_delivery"] = aggregate_response.get("duplicate_delivery")
    execution["supersedes_request_id"] = aggregate_response.get("supersedes_request_id")
    if not aggregate_response.get("ok"):
        return {"ok": False, "error": aggregate_response, "execution": execution}
    if aggregate_response.get("shard_id") != shard.shard_id:
        return {
            "ok": False,
            "error": make_error(
                "E_PROTOCOL", "node", node.node_id, "shard_id", "节点应答分片不匹配"
            ).to_dict(),
            "execution": execution,
        }
    return {
        "ok": True,
        "summary": aggregate_response["aggregate"],
        "node_id": node.node_id,
        "quota_version": state.active_quota_version,
        "width": width,
        "samples_fingerprint": samples_fingerprint(samples),
        "upstream_summary_fingerprints": upstream_summary_fingerprints,
        "execution": execution,
    }


def schedule_final_state(
    state: MaterializedState,
    manager: NodeProcessManager,
    *,
    previous_ledger: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    abort_after: int | None = None,
    mode: str = "incremental",
) -> dict[str, Any]:
    previous_ledger = previous_ledger or {}
    previous_cache = previous_ledger.get("cache", {})
    recovering = previous_ledger.get("phase") == "RUNNING"
    previous_results = previous_ledger.get("results", {}) if recovering else {}
    previous_counters = previous_ledger.get("counters", {}) if recovering else {}
    counters = SchedulerCounters(
        completed_cursor=list(previous_counters.get("completed_cursor", [])),
        pending=list(previous_counters.get("pending", [])),
        committed_units=int(previous_counters.get("committed_units", 0)),
        dispatch_units=int(previous_counters.get("dispatch_units", 0)),
        current_batch_index=int(previous_counters.get("current_batch_index", 0)),
        current_batch_points=int(previous_counters.get("current_batch_points", 0)),
        batch_plan=list(previous_counters.get("batch_plan", [])),
    )
    quota = state.active_quota
    quota_fp = quota.fingerprint
    cache: dict[str, Any] = {
        stage: dict(previous_cache.get(stage, {}))
        for stage in ("aggregate", "receipt", "watermark", "readiness")
    }
    results: dict[str, Any] = {
        key: dict(value) for key, value in previous_results.items()
    }
    errors: list[dict[str, Any]] = list(state.errors)
    if recovering:
        errors.extend(previous_ledger.get("errors", []))
    invalidation: list[dict[str, Any]] = []
    reuse: list[dict[str, Any]] = []
    execution_records: list[dict[str, Any]] = list(
        previous_ledger.get("execution_records", []) if recovering else []
    )
    observed_lease_chains: dict[str, Any] = dict(
        previous_ledger.get("lease_chains", {}) if recovering else {}
    )
    wait_records: list[dict[str, Any]] = [
        {
            "shard_id": shard_id,
            "blocked_by": sorted(state.shards[shard_id].upstream),
            "ready": False,
            "order": index,
        }
        for index, shard_id in enumerate(
            sorted(
                shard_id
                for shard_id, shard in state.shards.items()
                if shard.upstream
            )
        )
    ]
    downstream_fingerprints: dict[str, str] = {}
    downstream_artifact_tokens: dict[str, str] = {}
    batch_plan = counters.batch_plan
    order, topology_errors = _topological_order(state)
    errors.extend(topology_errors)
    reuse_plan, invalidation_plan = _content_resolution_plan(state)
    canonical_reuse, canonical_invalidation = _stage_resolution_ledgers(
        reuse_plan, invalidation_plan
    )

    def add_error(error: BizError | dict[str, Any]) -> None:
        errors.append(error.to_dict() if isinstance(error, BizError) else error)

    def make_checkpoint(phase: str = "RUNNING") -> None:
        for record in execution_records:
            node_id = record.get("node_id")
            generation = record.get("node_lease_generation")
            if node_id and generation is not None:
                observed_lease_chains[node_id] = {
                    "lease_generation": generation,
                    "last_lease_id": record.get("lease_id"),
                    "last_shard_id": record.get("shard_id"),
                }
        payload = {
            "phase": phase,
            "global_version": state.base_config.global_version,
            "base_config_hash": state.base_config_hash,
            "state_fingerprint": state.state_fingerprint,
            "event_stream_hash": fingerprint(
                [event.to_dict() for event in state.applied_events]
            ),
            "mode": mode,
            "results": results,
            "cache": cache,
            "counters": counters.to_dict(),
            "pending_queue": counters.pending,
            "completed_cursor": counters.completed_cursor,
            "invalidation_ledger": canonical_invalidation,
            "reuse_ledger": canonical_reuse,
            "cache_hit_ledger": reuse,
            "cache_miss_ledger": invalidation,
            "ready_wait_history": wait_records,
            "execution_records": execution_records,
            "errors": errors,
            "lease_chains": observed_lease_chains,
            "reuse_plan": reuse_plan,
            "invalidation_plan": invalidation_plan,
        }
        if checkpoint is not None:
            checkpoint(payload)

    for shard_id in order:
        if shard_id in results:
            summary = results[shard_id].get("aggregate", {}).get("summary", {})
            summary_fp = summary.get("summary_fingerprint", "")
            downstream_fingerprints[shard_id] = summary_fp
            downstream_artifact_tokens[shard_id] = fingerprint(
                {
                    "summary_fingerprint": summary_fp,
                    "aggregate_cache_key": results[shard_id]
                    .get("cache_keys", {})
                    .get("aggregate"),
                }
            )
            continue
        shard = state.shards[shard_id]
        shard_fp = samples_fingerprint(state.samples(shard_id))
        blockers = [
            dependency
            for dependency in sorted(shard.upstream)
            if dependency not in downstream_fingerprints
        ]
        upstream_fps = [
            downstream_fingerprints[dependency]
            for dependency in sorted(shard.upstream)
            if dependency in downstream_fingerprints
        ]
        upstream_tokens = [
            downstream_artifact_tokens[dependency]
            for dependency in sorted(shard.upstream)
            if dependency in downstream_artifact_tokens
        ]
        logical_generation = int(state.shard_revisions.get(shard_id, 1))
        effective_node = state.effective_node(shard.node_id)
        units = _resource_units(shard, quota.points_per_resource_unit)
        points = _shard_points(shard)
        watermark = evaluate_watermark(
            effective_node.capacity_slots,
            units,
            quota.soft_watermark_bp,
            quota.hard_watermark_bp,
            state.width,
        )
        wm_key = watermark_key(
            node_id=shard.node_id,
            shard_id=shard_id,
            quota_fingerprint=quota_fp,
            width=state.width,
            resource_units=units,
            capacity_slots=effective_node.capacity_slots,
            lease_generation=logical_generation,
        )
        rd_key = readiness_key(
            shard_id=shard_id,
            upstream=shard.upstream,
            upstream_summary_fingerprints=upstream_fps,
            quota_fingerprint=quota_fp,
            lease_generation=logical_generation,
            upstream_artifact_tokens=upstream_tokens,
        )
        agg_key = aggregate_key(
            node_id=shard.node_id,
            shard_id=shard_id,
            samples_fingerprint=shard_fp,
            quota_fingerprint=quota_fp,
            width=state.width,
            lease_generation=logical_generation,
            upstream_summary_fingerprints=upstream_fps,
            upstream_artifact_tokens=upstream_tokens,
        )
        readiness_product = {
            "ready": not blockers,
            "blockers": blockers,
            "upstream_fingerprints": upstream_fps,
        }
        if rd_key in cache["readiness"]:
            readiness_product = cache["readiness"][rd_key]["product"]
            reuse.append(
                {
                    "stage": "readiness",
                    "shard_id": shard_id,
                    "cache_key": rd_key,
                    "basis": "upstream,upstream_fingerprints,quota,lease_generation unchanged",
                }
            )
        else:
            prior = _cache_records_for_shard(previous_cache, "readiness", shard_id)
            reason = _first_invalidation_reason(prior, {
                "cache_key": rd_key,
                "shard_id": shard_id,
                "upstream": sorted(shard.upstream),
                "quota_version": state.active_quota_version,
                "quota_fingerprint": quota_fp,
                "width": state.width,
                "lease_generation": logical_generation,
            })
            invalidation.append(
                {"stage": "readiness", "shard_id": shard_id, "cache_key": rd_key, **reason}
            )
            cache["readiness"][rd_key] = {
                "cache_key": rd_key,
                "shard_id": shard_id,
                "upstream": sorted(shard.upstream),
                "quota_version": state.active_quota_version,
                "quota_fingerprint": quota_fp,
                "width": state.width,
                "lease_generation": logical_generation,
                "product": readiness_product,
            }
        if blockers:
            add_error(
                make_error(
                    "E_DEPENDENCY_BLOCKED",
                    "shard",
                    shard_id,
                    "upstream",
                    "上游尚未产出权威聚合摘要",
                    details={"blockers": blockers},
                )
            )
            counters.pending.append(
                {
                    "shard_id": shard_id,
                    "blockers": blockers,
                    "reason": "E_DEPENDENCY_BLOCKED",
                }
            )
            results[shard_id] = {
                "shard_id": shard_id,
                "ready": readiness_product,
                "aggregate": None,
                "receipt": None,
                "watermark": watermark,
                "cache_keys": {
                    "readiness": rd_key,
                    "watermark": wm_key,
                    "aggregate": agg_key,
                },
                "blocked": True,
            }
            make_checkpoint()
            continue
        current_wm = {
            "cache_key": wm_key,
            "shard_id": shard_id,
            "upstream": sorted(shard.upstream),
            "quota_version": state.active_quota_version,
            "quota_fingerprint": quota_fp,
            "width": state.width,
            "lease_generation": logical_generation,
            "upstream_artifact_tokens": upstream_tokens,
            "product": watermark,
        }
        if wm_key in cache["watermark"]:
            watermark = cache["watermark"][wm_key]["product"]
            reuse.append(
                {
                    "stage": "watermark",
                    "shard_id": shard_id,
                    "cache_key": wm_key,
                    "basis": "quota_fingerprint,width,resource_units,capacity_slots,lease_generation unchanged",
                }
            )
        else:
            previous = _cache_records_for_shard(previous_cache, "watermark", shard_id)
            reason = _first_invalidation_reason(previous, current_wm)
            invalidation.append(
                {"stage": "watermark", "shard_id": shard_id, "cache_key": wm_key, **reason}
            )
            cache["watermark"][wm_key] = current_wm
        if watermark["code"]:
            add_error(
                make_error(
                    watermark["code"],
                    "shard",
                    shard_id,
                    "watermark",
                    watermark["state"],
                    details=watermark,
                )
            )
        if watermark["state"] == "HARD_REJECT":
            counters.pending.append(
                {
                    "shard_id": shard_id,
                    "blockers": [],
                    "reason": "E_QUOTA_HARD_LIMIT",
                }
            )
            results[shard_id] = {
                "shard_id": shard_id,
                "ready": readiness_product,
                "aggregate": None,
                "receipt": None,
                "watermark": watermark,
                "cache_keys": {
                    "readiness": rd_key,
                    "watermark": wm_key,
                    "aggregate": agg_key,
                },
                "blocked": True,
            }
            make_checkpoint()
            continue
        if points > state.base_config.max_batch_points:
            add_error(
                make_error(
                    "E_BATCH_POINTS_EXCEEDED",
                    "shard",
                    shard_id,
                    "batches",
                    "单分片处理点数超过单批上限",
                    details={"points": points, "max_batch_points": state.base_config.max_batch_points},
                )
            )
            counters.pending.append(
                {"shard_id": shard_id, "blockers": [], "reason": "E_BATCH_POINTS_EXCEEDED"}
            )
            results[shard_id] = {
                "shard_id": shard_id,
                "ready": readiness_product,
                "aggregate": None,
                "receipt": None,
                "watermark": watermark,
                "cache_keys": {
                    "readiness": rd_key,
                    "watermark": wm_key,
                    "aggregate": agg_key,
                },
                "blocked": True,
            }
            make_checkpoint()
            continue
        if counters.committed_units + units > state.base_config.resource_budget:
            add_error(
                make_error(
                    "E_GLOBAL_BUDGET_EXHAUSTED",
                    "scheduler",
                    shard_id,
                    "resource_budget",
                    "全局资源单位预算不足",
                    details={
                        "committed_units": counters.committed_units,
                        "required_units": units,
                        "resource_budget": state.base_config.resource_budget,
                    },
                )
            )
            counters.pending.append(
                {"shard_id": shard_id, "blockers": [], "reason": "E_GLOBAL_BUDGET_EXHAUSTED"}
            )
            results[shard_id] = {
                "shard_id": shard_id,
                "ready": readiness_product,
                "aggregate": None,
                "receipt": None,
                "watermark": watermark,
                "cache_keys": {
                    "readiness": rd_key,
                    "watermark": wm_key,
                    "aggregate": agg_key,
                },
                "blocked": True,
            }
            make_checkpoint()
            continue
        if counters.current_batch_points and (
            counters.current_batch_points + points > state.base_config.max_batch_points
        ):
            counters.current_batch_index += 1
            counters.current_batch_points = 0
            batch_plan.append(
                {
                    "batch_index": counters.current_batch_index,
                    "shard_ids": [],
                    "points": 0,
                }
            )
        if not batch_plan or batch_plan[-1]["batch_index"] != counters.current_batch_index:
            batch_plan.append(
                {
                    "batch_index": counters.current_batch_index,
                    "shard_ids": [],
                    "points": 0,
                }
            )
        batch_plan[-1]["shard_ids"].append(shard_id)
        batch_plan[-1]["points"] += points
        counters.current_batch_points += points
        counters.committed_units += units
        current_agg = {
            "cache_key": agg_key,
            "shard_id": shard_id,
            "upstream": sorted(shard.upstream),
            "quota_version": state.active_quota_version,
            "quota_fingerprint": quota_fp,
            "width": state.width,
            "lease_generation": logical_generation,
            "upstream_artifact_tokens": upstream_tokens,
        }
        aggregate_product: dict[str, Any] | None = None
        if agg_key in cache["aggregate"]:
            aggregate_product = cache["aggregate"][agg_key]["product"]
            reuse.append(
                {
                    "stage": "aggregate",
                    "shard_id": shard_id,
                    "cache_key": agg_key,
                    "basis": "samples,upstream,quota,width,lease_generation unchanged",
                }
            )
        else:
            previous = _cache_records_for_shard(previous_cache, "aggregate", shard_id)
            reason = _first_invalidation_reason(previous, current_agg)
            invalidation.append(
                {"stage": "aggregate", "shard_id": shard_id, "cache_key": agg_key, **reason}
            )
            execution_result = _execute_shard(
                state=state,
                manager=manager,
                shard=shard,
                quota_fingerprint=quota_fp,
                width=state.width,
                lease_generation=logical_generation,
                upstream_summary_fingerprints=upstream_fps,
            )
            execution_records.append(
                {
                    "shard_id": shard_id,
                    "node_id": shard.node_id,
                    **execution_result.get("execution", {}),
                }
            )
            if not execution_result.get("ok"):
                error = execution_result.get("error", {})
                if "code" in error:
                    add_error(error)
                else:
                    add_error(
                        make_error(
                            "E_NODE_UNAVAILABLE",
                            "shard",
                            shard_id,
                            "node",
                            "节点聚合执行失败",
                        )
                    )
                counters.pending.append(
                    {"shard_id": shard_id, "blockers": [], "reason": "NODE_EXECUTION_FAILED"}
                )
                results[shard_id] = {
                    "shard_id": shard_id,
                    "ready": readiness_product,
                    "aggregate": None,
                    "receipt": None,
                    "watermark": watermark,
                    "cache_keys": {
                        "readiness": rd_key,
                        "watermark": wm_key,
                        "aggregate": agg_key,
                    },
                    "blocked": True,
                }
                make_checkpoint()
                continue
            aggregate_product = {
                "summary": execution_result["summary"],
                "node_id": execution_result["node_id"],
                "quota_version": state.active_quota_version,
                "width": state.width,
                "samples_fingerprint": execution_result["samples_fingerprint"],
            "upstream_summary_fingerprints": upstream_fps,
            "upstream_artifact_tokens": upstream_tokens,
            }
            counters.dispatch_units += units
            cache["aggregate"][agg_key] = {
                **current_agg,
                "product": aggregate_product,
            }
        summary = aggregate_product["summary"]
        summary_fp = summary["summary_fingerprint"]
        downstream_fingerprints[shard_id] = summary_fp
        downstream_artifact_tokens[shard_id] = fingerprint(
            {"summary_fingerprint": summary_fp, "aggregate_cache_key": agg_key}
        )
        receipt_cache_key = receipt_key(
            shard_id=shard_id,
            aggregate_summary_fingerprint=summary_fp,
            quota_fingerprint=quota_fp,
            width=state.width,
            lease_generation=logical_generation,
            upstream_summary_fingerprints=upstream_fps,
            upstream_artifact_tokens=upstream_tokens,
        )
        current_receipt = {
            "cache_key": receipt_cache_key,
            "shard_id": shard_id,
            "upstream": sorted(shard.upstream),
            "quota_version": state.active_quota_version,
            "quota_fingerprint": quota_fp,
            "width": state.width,
            "lease_generation": logical_generation,
            "upstream_artifact_tokens": upstream_tokens,
        }
        if receipt_cache_key in cache["receipt"]:
            receipt = cache["receipt"][receipt_cache_key]["product"]
            reuse.append(
                {
                    "stage": "receipt",
                    "shard_id": shard_id,
                    "cache_key": receipt_cache_key,
                    "basis": "aggregate_summary,quota,width,lease_generation unchanged",
                }
            )
        else:
            prior = _cache_records_for_shard(previous_cache, "receipt", shard_id)
            reason = _first_invalidation_reason(prior, current_receipt)
            invalidation.append(
                {"stage": "receipt", "shard_id": shard_id, "cache_key": receipt_cache_key, **reason}
            )
            receipt = {
                "authoritative": True,
                "shard_id": shard_id,
                "node_id": aggregate_product["node_id"],
                "quota_version": state.active_quota_version,
                "quota_fingerprint": quota_fp,
                "width": state.width,
                "lease_generation": logical_generation,
                "aggregate_summary_fingerprint": summary_fp,
                "upstream": sorted(shard.upstream),
                "upstream_summary_fingerprints": sorted(upstream_fps),
                "cache_key": receipt_cache_key,
            }
            cache["receipt"][receipt_cache_key] = {
                **current_receipt,
                "product": receipt,
            }
        results[shard_id] = {
            "shard_id": shard_id,
            "ready": readiness_product,
            "aggregate": {
                "summary": summary,
                "samples_fingerprint": aggregate_product["samples_fingerprint"],
                "node_id": aggregate_product["node_id"],
            },
            "receipt": receipt,
            "watermark": watermark,
            "cache_keys": {
                "readiness": rd_key,
                "watermark": wm_key,
                "aggregate": agg_key,
                "receipt": receipt_cache_key,
            },
            "blocked": False,
        }
        counters.completed_cursor.append(shard_id)
        counters.pending = [
            entry for entry in counters.pending if entry.get("shard_id") != shard_id
        ]
        make_checkpoint()
        if abort_after is not None and len(counters.completed_cursor) >= abort_after:
            raise ForcedTermination(len(counters.completed_cursor))

    make_checkpoint("COMPLETE")
    runtime_nodes = manager.statuses()
    node_ledgers = manager.ledgers()
    leases = {
        node_id: {
            "lease_generation": status.get("lease_generation"),
            "active_work_units": status.get("active_work_units"),
            "errors_count": status.get("errors_count"),
        }
        for node_id, status in runtime_nodes.items()
    }
    unique_errors: dict[str, dict[str, Any]] = {}
    for error in errors:
        unique_errors.setdefault(fingerprint(error), error)
    errors_sorted = sorted(
        list(unique_errors.values()),
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("object_type", "")),
            str(item.get("object_id", "")),
            str(item.get("field", "")),
        ),
    )
    error_counts: dict[str, int] = {}
    for error in errors_sorted:
        code = str(error.get("code", "E_INTERNAL"))
        error_counts[code] = error_counts.get(code, 0) + 1
    result = {
        "schema_version": "shard-diagnostic-v1",
        "metadata": {
            "global_version": state.base_config.global_version,
            "base_config_hash": state.base_config_hash,
            "state_fingerprint": state.state_fingerprint,
            "event_stream_hash": fingerprint(
                [event.to_dict() for event in state.applied_events]
            ),
            "seed": state.base_config.seed,
            "active_quota_version": state.active_quota_version,
            "quota_fingerprint": quota_fp,
            "width": state.width,
            "reference_width": state.base_config.reference_width,
            "points_per_resource_unit": quota.points_per_resource_unit,
            "fixed_scale": 1000,
            "rounding_policy": ROUNDING_POLICY,
            "lease_ttl_ms": state.base_config.lease_ttl_ms,
            "deterministic_order": order,
            "soft_boundary_rule": "resource_units > soft_limit triggers warning",
            "hard_boundary_rule": "resource_units > hard_limit rejects dispatch",
        },
        "quota_versions": [
            {
                "version": version,
                "parent_version": state.quotas[version].parent_version,
                "fingerprint": state.quotas[version].fingerprint,
            }
            for version in sorted(state.quotas)
        ],
        "nodes": [
            state.effective_node(node_id).to_dict() for node_id in sorted(state.nodes)
        ],
        "per_shard": {
            shard_id: results[shard_id] for shard_id in sorted(results)
        },
        "ready_wait_history": sorted(wait_records, key=lambda item: int(item.get("order", 0))),
        "ready_wait_queue": [
            entry
            for entry in counters.pending
            if entry.get("reason") == "E_DEPENDENCY_BLOCKED"
        ],
        "watermarks": {
            shard_id: results[shard_id].get("watermark")
            for shard_id in sorted(results)
        },
        "reuse_ledger": canonical_reuse,
        "invalidation_ledger": canonical_invalidation,
        "cache_hit_ledger": reuse,
        "cache_miss_ledger": invalidation,
        "reuse_plan": reuse_plan,
        "invalidation_plan": invalidation_plan,
        "errors": errors_sorted,
        "error_counts": dict(sorted(error_counts.items())),
        "budget": {
            "resource_budget": state.base_config.resource_budget,
            "committed_units": counters.committed_units,
            "dispatch_units": counters.dispatch_units,
            "remaining_units": state.base_config.resource_budget - counters.committed_units,
        },
        "batch_plan": sorted(batch_plan, key=lambda item: int(item.get("batch_index", 0))),
        "completed_cursor": list(counters.completed_cursor),
        "pending_queue": list(counters.pending),
        "cache": cache,
        "results": results,
        "counters": counters.to_dict(),
        "execution_records": execution_records,
        "execution": {
            "mode": mode,
            "runtime_nodes": runtime_nodes,
            "lease_chains": leases,
            "node_ledgers": node_ledgers,
            "recovered_from_snapshot": previous_ledger.get("_snapshot_manifest"),
        },
    }
    return result


def run_pipeline(
    config: ProjectConfig,
    state: MaterializedState,
    workdir: Path,
    *,
    mode: str = "incremental",
    use_previous: bool = True,
    abort_after: int | None = None,
    checkpoint_on_each_shard: bool = True,
) -> dict[str, Any]:
    store = SnapshotStore(
        workdir,
        global_version=config.global_version,
        base_config_hash=config.base_config_hash,
    )
    previous: dict[str, Any] | None = None
    snapshot_status: dict[str, Any] = {}
    if use_previous:
        try:
            previous = store.load()
        except Exception as exc:
            if hasattr(exc, "reason"):
                snapshot_status = {
                    "loaded": False,
                    "version_change": getattr(exc, "reason", str(exc)),
                }
                previous = None
            else:
                raise
    if previous is not None:
        snapshot_status = {
            "loaded": True,
            "termination": termination_classification(previous),
            "manifest": previous.get("_snapshot_manifest"),
        }
    else:
        snapshot_status.setdefault("loaded", False)

    manager = NodeProcessManager(
        [state.effective_node(node_id) for node_id in sorted(state.nodes)],
        workdir / "nodes",
    )
    commit_count = 0

    def commit(ledger_payload: dict[str, Any]) -> None:
        nonlocal commit_count
        manifest = store.commit(ledger_payload)
        commit_count += 1
        ledger_payload["_snapshot_manifest"] = manifest

    result: dict[str, Any]
    try:
        manager.start_all()
        result = schedule_final_state(
            state,
            manager,
            previous_ledger=previous,
            checkpoint=commit if checkpoint_on_each_shard else None,
            abort_after=abort_after,
            mode=mode,
        )
    finally:
        manager.stop_all()
    if abort_after is None:
        result["phase"] = "COMPLETE"
        manifest = store.commit(result)
        result["execution"]["final_snapshot"] = manifest
    result["execution"]["snapshot_status"] = snapshot_status
    result["execution"]["snapshot_commits"] = commit_count
    return result

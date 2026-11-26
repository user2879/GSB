from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .fixedpoint import aggregate_groups, evaluate_watermark
from .generator import generate_shard_samples
from .models import ProjectConfig, QuotaVersion
from .node import NodeRuntime
from .state import MaterializedState
from .validation import validate_config


def _validation_injection(
    config: ProjectConfig,
    injection_id: str,
    kind: str,
) -> dict[str, Any]:
    raw = config.to_dict()
    raw["events"] = []
    raw["injections"] = []
    if kind == "dependency_cycle":
        shards = raw.get("shards", [])
        if len(shards) < 2:
            return {
                "injection_id": injection_id,
                "kind": kind,
                "hit": False,
                "codes": [],
                "error": "至少需要两个分片才能制造依赖环",
            }
        first, second = shards[0], shards[1]
        first["upstream"] = [second["shard_id"]]
        second["upstream"] = [first["shard_id"]]
        injected = ProjectConfig.from_raw(raw)
        report = validate_config(injected)
    elif kind == "quota_missing_parent":
        quotas = raw.get("quota_versions", [])
        if not quotas:
            return {
                "injection_id": injection_id,
                "kind": kind,
                "hit": False,
                "codes": [],
                "error": "没有配额版本可扩展",
            }
        template = copy.deepcopy(quotas[-1])
        template["version"] = max(int(item["version"]) for item in quotas) + 1
        template["parent_version"] = 9999
        quotas.append(template)
        injected = ProjectConfig.from_raw(raw)
        report = validate_config(injected)
    else:
        return {
            "injection_id": injection_id,
            "kind": kind,
            "hit": False,
            "codes": [],
            "error": "未知验证注入",
        }
    codes = [error.code for error in report.errors]
    return {
        "injection_id": injection_id,
        "kind": kind,
        "hit": bool(codes),
        "codes": codes,
        "errors": [error.to_dict() for error in report.errors],
        "isolated": True,
    }


def _sample_dimension_injection(state: MaterializedState) -> dict[str, Any]:
    node_id = sorted(state.nodes)[0]
    spec = state.effective_node(node_id)
    capacity = max(spec.capacity_slots, spec.lease_limit, 2)
    spec = type(spec)(
        node_id=spec.node_id,
        host=spec.host,
        port=spec.port,
        capacity_slots=capacity,
        lease_limit=max(spec.lease_limit, 2),
    )
    runtime = NodeRuntime(spec=spec)
    acquire = runtime.acquire({"shard_id": "inject-dim-a", "ttl_ms": 1000})
    runtime.aggregate(
        {
            "request_id": "inject-dim-a",
            "shard_id": "inject-dim-a",
            "batch_id": "b",
            "quota_version": state.active_quota_version,
            "width": state.width,
            "dimensions": 3,
            "samples": [{"group": "g0", "values": [1, 2]}],
            "lease_id": acquire["lease_id"],
            "expected_generation": acquire["lease_generation"],
        }
    )
    runtime.release(
        {
            "lease_id": acquire["lease_id"],
            "expected_generation": acquire["lease_generation"],
        }
    )
    second = runtime.acquire({"shard_id": "inject-dim-b", "ttl_ms": 1000})
    valid = runtime.aggregate(
        {
            "request_id": "inject-dim-b",
            "shard_id": "inject-dim-b",
            "batch_id": "b",
            "quota_version": state.active_quota_version,
            "width": state.width,
            "dimensions": 2,
            "samples": [{"group": "g0", "values": [4, 5]}],
            "lease_id": second["lease_id"],
            "expected_generation": second["lease_generation"],
        }
    )
    runtime.release(
        {
            "lease_id": second["lease_id"],
            "expected_generation": second["lease_generation"],
        }
    )
    codes = [error.code for error in runtime.errors]
    return {
        "injection_id": "I_SAMPLE_DIMENSION",
        "kind": "sample_dimension",
        "hit": "E_SAMPLE_DIMENSION" in codes,
        "codes": codes,
        "other_shard_succeeded": bool(valid.get("ok")),
        "isolated": bool(valid.get("ok")),
    }


def _stale_lease_injection(state: MaterializedState) -> dict[str, Any]:
    node_id = sorted(state.nodes)[0]
    spec = state.effective_node(node_id)
    runtime = NodeRuntime(spec=spec)
    acquire = runtime.acquire({"shard_id": "stale-lease", "ttl_ms": 1000})
    generation = acquire["lease_generation"]
    stale = runtime.renew(
        {
            "lease_id": acquire["lease_id"],
            "expected_generation": generation - 1,
            "ttl_ms": 1000,
        }
    )
    generation_unchanged = runtime.lease_generation == generation
    assert stale["code"] == "E_LEASE_GENERATION_MISMATCH"
    release = runtime.release(
        {
            "lease_id": acquire["lease_id"],
            "expected_generation": generation,
        }
    )
    codes = [error.code for error in runtime.errors]
    return {
        "injection_id": "I_STALE_LEASE",
        "kind": "stale_lease",
        "hit": "E_LEASE_GENERATION_MISMATCH" in codes,
        "codes": codes,
        "generation_unchanged": generation_unchanged,
        "valid_release_after_reject": bool(release.get("ok")),
        "isolated": generation_unchanged and bool(release.get("ok")),
    }


def _expired_renew_injection(state: MaterializedState) -> dict[str, Any]:
    node_id = sorted(state.nodes)[0]
    spec = state.effective_node(node_id)
    now = [1000]

    def clock() -> int:
        return now[0]

    runtime = NodeRuntime(spec=spec, clock_ms=clock)
    acquire = runtime.acquire({"shard_id": "expired-lease", "ttl_ms": 5})
    generation = acquire["lease_generation"]
    now[0] += 10
    sweep = runtime.sweep()
    renew = runtime.renew(
        {
            "lease_id": acquire["lease_id"],
            "expected_generation": generation,
            "ttl_ms": 5,
        }
    )
    reused = runtime.acquire(
        {"shard_id": "expired-lease-reuse", "ttl_ms": 50, "expected_generation": runtime.lease_generation}
    )
    codes = [error.code for error in runtime.errors]
    return {
        "injection_id": "I_EXPIRED_RENEW",
        "kind": "expired_renew",
        "hit": "E_LEASE_EXPIRED" in codes,
        "codes": codes,
        "expired_count": len(sweep.get("expired", [])),
        "active_after_sweep": sweep.get("active_work_units"),
        "slot_reused": bool(reused.get("ok")),
        "isolated": bool(renew.get("ok") is False and reused.get("ok")),
    }


def run_injections(state: MaterializedState) -> list[dict[str, Any]]:
    return [
        _validation_injection(state.base_config, "I_DEPENDENCY_CYCLE", "dependency_cycle"),
        _validation_injection(
            state.base_config, "I_QUOTA_PARENT_MISSING", "quota_missing_parent"
        ),
        _sample_dimension_injection(state),
        _stale_lease_injection(state),
        _expired_renew_injection(state),
    ]


def _aggregate_attribution(
    shard_id: str,
    state: MaterializedState,
    active_summary: dict[str, Any] | None,
    reference_width: int,
) -> dict[str, Any]:
    shard = state.shards[shard_id]
    samples = generate_shard_samples(state.base_config.seed, shard)
    reference = aggregate_groups(samples, shard.dimensions, reference_width, 1000)
    if active_summary is None:
        return {
            "active_summary_fingerprint": None,
            "reference_summary_fingerprint": reference["summary_fingerprint"],
            "different": True,
            "first_group": None,
            "first_dimension": None,
            "delta": None,
            "differing_dimensions": 0,
        }
    active_groups = {group["group"]: group for group in active_summary.get("groups", [])}
    reference_groups = {group["group"]: group for group in reference.get("groups", [])}
    differing_dimensions = 0
    first_group: str | None = None
    first_dimension: int | None = None
    first_delta: int | None = None
    for key in sorted(set(reference_groups) | set(active_groups)):
        active_group = active_groups.get(key)
        reference_group = reference_groups.get(key)
        if active_group is None or reference_group is None:
            differing_dimensions += max(
                len(active_group.get("quantized_sum", [])) if active_group else 0,
                len(reference_group.get("quantized_sum", [])) if reference_group else 0,
            )
            if first_group is None:
                first_group = key
                first_dimension = 0
            continue
        for dimension in range(shard.dimensions):
            active_value = int(active_group["quantized_sum"][dimension])
            reference_value = int(reference_group["quantized_sum"][dimension])
            if active_value != reference_value:
                differing_dimensions += 1
                if first_group is None:
                    first_group = key
                    first_dimension = dimension
                    first_delta = active_value - reference_value
    return {
        "active_summary_fingerprint": active_summary.get("summary_fingerprint"),
        "reference_summary_fingerprint": reference["summary_fingerprint"],
        "different": differing_dimensions > 0,
        "first_group": first_group,
        "first_dimension": first_dimension,
        "delta": first_delta,
        "differing_dimensions": differing_dimensions,
    }


def build_diagnostic_report(
    state: MaterializedState,
    result: dict[str, Any],
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    quota: QuotaVersion = state.active_quota
    reference_width = min(state.width + 1, 62)
    attribution: dict[str, Any] = {}
    sources: Counter[str] = Counter()
    for shard_id in sorted(state.shards):
        shard = state.shards[shard_id]
        shard_result = result.get("per_shard", {}).get(shard_id, {})
        aggregate = shard_result.get("aggregate")
        active_summary = aggregate.get("summary") if aggregate else None
        agg_diff = _aggregate_attribution(
            shard_id, state, active_summary, reference_width
        )
        active_watermark = shard_result.get("watermark") or evaluate_watermark(
            state.effective_node(shard.node_id).capacity_slots,
            max(1, sum(b.sample_count * shard.dimensions for b in shard.batches) // quota.points_per_resource_unit),
            quota.soft_watermark_bp,
            quota.hard_watermark_bp,
            state.width,
        )
        reference_watermark = evaluate_watermark(
            state.effective_node(shard.node_id).capacity_slots,
            int(active_watermark.get("resource_units", 0)),
            quota.soft_watermark_bp,
            quota.hard_watermark_bp,
            reference_width,
        )
        watermark_diff = {
            "different": active_watermark.get("soft_limit") != reference_watermark.get("soft_limit")
            or active_watermark.get("hard_limit") != reference_watermark.get("hard_limit"),
            "active_soft_limit": active_watermark.get("soft_limit"),
            "reference_soft_limit": reference_watermark.get("soft_limit"),
            "active_hard_limit": active_watermark.get("hard_limit"),
            "reference_hard_limit": reference_watermark.get("hard_limit"),
            "first_exceed_stage": "watermark_rounding"
            if active_watermark.get("soft_limit") != reference_watermark.get("soft_limit")
            or active_watermark.get("hard_limit") != reference_watermark.get("hard_limit")
            else None,
        }
        readiness_active = shard_result.get("ready", {})
        expected_ready = all(
            result.get("per_shard", {}).get(dependency, {}).get("receipt") is not None
            for dependency in shard.upstream
        )
        readiness_diff = {
            "active_ready": readiness_active.get("ready"),
            "reference_ready": expected_ready,
            "different": bool(readiness_active.get("ready")) != expected_ready,
            "blockers": readiness_active.get("blockers", []),
        }
        first_stage = "NONE"
        if agg_diff["different"]:
            first_stage = "aggregate_quantization"
            sources["aggregate_rounding"] += 1
        elif watermark_diff["different"]:
            first_stage = "watermark_rounding"
            sources["watermark_rounding"] += 1
        elif readiness_diff["different"]:
            first_stage = "dependency_decision"
            sources["dependency_chain_change"] += 1
        attribution[shard_id] = {
            "reference_width": reference_width,
            "configured_reference_width": state.base_config.reference_width,
            "active_width": state.width,
            "aggregate_quantization": agg_diff,
            "watermark_rounding": watermark_diff,
            "dependency_decision": readiness_diff,
            "first_exceed_stage": first_stage,
        }
    wait_by_shard = {
        entry["shard_id"]: len(entry.get("blocked_by", []))
        for entry in result.get("ready_wait_history", [])
    }
    wait_distribution = Counter(str(value) for value in wait_by_shard.values())
    injections = run_injections(state)
    injection_hits = [entry["injection_id"] for entry in injections if entry.get("hit")]
    node_lease_generations = {
        node_id: info.get("lease_generation")
        for node_id, info in sorted(
            result.get("execution", {}).get("lease_chains", {}).items()
        )
    }
    report = {
        "schema_version": "shard-diagnostic-report-v1",
        "metadata": result.get("metadata", {}),
        "run_state_fingerprint": state.state_fingerprint,
        "nodes": result.get("nodes", []),
        "node_lease_generations": node_lease_generations,
        "ready_wait_distribution": {
            "blocked_shard_count": len(wait_by_shard),
            "blocker_count_buckets": dict(sorted(wait_distribution.items())),
            "by_shard": dict(sorted(wait_by_shard.items())),
        },
        "error_counts": result.get("error_counts", {}),
        "injections": injections,
        "injection_hits": injection_hits,
        "fixedpoint_attribution": attribution,
        "conclusion_consistency": {
            "different_shards": sorted(
                shard_id
                for shard_id, item in attribution.items()
                if item["first_exceed_stage"] != "NONE"
            ),
            "sources": dict(sorted(sources.items())),
            "reference_width": reference_width,
            "active_width": state.width,
        },
        "replay_fingerprint": fingerprint(
            {
                "per_shard": {
                    shard_id: {
                        "aggregate": result.get("per_shard", {}).get(shard_id, {}).get("aggregate"),
                        "receipt": result.get("per_shard", {}).get(shard_id, {}).get("receipt"),
                        "watermark": result.get("per_shard", {}).get(shard_id, {}).get("watermark"),
                    }
                    for shard_id in sorted(state.shards)
                },
                "ready_wait_history": result.get("ready_wait_history", []),
            }
        ),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    return report

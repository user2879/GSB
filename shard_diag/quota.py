from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .canonical import saturate_int64
from .graph import stable_topology


def resolve_quota(versions: list[dict[str, Any]], target_version: int) -> dict[str, Any]:
    by_version = {item["version"]: item for item in versions}
    if target_version not in by_version:
        from .errors import SpecError

        raise SpecError("目标配额版本不存在", entry="quota_versions.json", field="version")

    state: dict[str, Any] = {
        "version": 0,
        "parent_version": None,
        "session_scale": 1,
        "global_budget_units": 0,
        "node_quotas": {},
        "batch_requirements": {},
        "scale_changed_in_target": False,
    }
    for version_number in range(1, target_version + 1):
        change = by_version[version_number]
        state["version"] = version_number
        state["parent_version"] = change["parent_version"]
        state["scale_changed_in_target"] = False
        if change["session_scale"] is not None:
            if version_number == target_version and state["session_scale"] != change["session_scale"]:
                state["scale_changed_in_target"] = version_number != 1
            state["session_scale"] = int(change["session_scale"])
        if change["global_budget_units"] is not None:
            state["global_budget_units"] = int(change["global_budget_units"])
        for node_id, quota in change["upsert_node_quotas"].items():
            state["node_quotas"][node_id] = deepcopy(quota)
        for node_id in change["delete_node_quotas"]:
            state["node_quotas"].pop(node_id, None)
        for key, units in change["upsert_batch_requirements"].items():
            state["batch_requirements"][key] = int(units)
        for key in change["delete_batch_requirements"]:
            state["batch_requirements"].pop(key, None)
    return state


def convert_fixed_scale(value_q: int, old_scale: int, new_scale: int) -> dict[str, int | bool]:
    """Convert a non-negative fixed-point integer with floor truncation."""

    if old_scale <= 0 or new_scale <= 0:
        raise ValueError("scales must be positive")
    numerator = int(value_q) * new_scale
    converted = numerator // old_scale
    remainder_numerator = numerator - converted * old_scale
    return {
        "value_q": saturate_int64(converted),
        "floor_remainder_numerator": remainder_numerator,
        "error_bound_q": 1,
        "overflow_saturated": saturate_int64(converted) != converted,
        "truncation": "nonnegative_floor",
    }


def q(units: int, scale: int) -> int:
    return saturate_int64(max(0, int(units)) * int(scale))


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    magnitude = abs(int(numerator)) // abs(int(denominator))
    same_sign = (int(numerator) < 0) == (int(denominator) < 0)
    return magnitude if same_sign else -magnitude


def batch_accumulation_error_q(resource_units: int, batch_count: int, scale: int) -> int:
    """Conservative upper bound in fixed-point q units.

    Splitting ``U*S`` into B floored batch quotas loses at most B-1 q units;
    the exact remainder is ``(U*S) mod B`` when callers need a tighter value.
    """
    if batch_count <= 1:
        return 0
    _ = int(resource_units), int(scale)
    return int(batch_count) - 1


def batch_accumulation_remainder_q(resource_units: int, batch_count: int, scale: int) -> int:
    if batch_count <= 1:
        return 0
    return (max(0, int(resource_units)) * int(scale)) % int(batch_count)


def _comparison_error_q(capacity_q: int, scale: int) -> int:
    if capacity_q <= 0 or scale <= 0:
        return 0
    return int(math.ceil(capacity_q / scale))


def quota_error_breakdown(
    *,
    scope: str,
    capacity_q: int,
    scale: int,
    resource_units: int,
    batch_count: int,
    baseline_scopes: int = 1,
) -> dict[str, int | str]:
    baseline_error = max(0, int(baseline_scopes))
    shard_error = batch_accumulation_error_q(resource_units, max(1, batch_count), scale)
    comparison_error = _comparison_error_q(capacity_q, scale)
    total = baseline_error + shard_error + comparison_error
    return {
        "scope": scope,
        "quota_baseline_error_q": baseline_error,
        "shard_accumulation_error_q": shard_error,
        "node_comparison_error_q": comparison_error,
        "synthetic_error_bound_q": total,
        "unit": "fixed_point_q_ulps",
    }


def effective_node_quota(
    node: dict[str, Any],
    resolved: dict[str, Any],
) -> dict[str, int]:
    declared = {
        "concurrency": int(node["concurrency"]),
        "lease_limit": int(node["lease_limit"]),
        "resource_units_cap": 0,
    }
    version_quota = resolved["node_quotas"].get(node["node_id"])
    if not version_quota:
        return {**declared, "concurrency": 0, "lease_limit": 0, "resource_units_cap": 0}
    return {
        "concurrency": min(declared["concurrency"], int(version_quota["concurrency"])),
        "lease_limit": min(declared["lease_limit"], int(version_quota["lease_limit"])),
        "resource_units_cap": int(version_quota["resource_units_cap"]),
    }


def compute_water(
    spec_nodes: list[dict[str, Any]],
    resolved: dict[str, Any],
    active_usage: dict[str, dict[str, int]],
    *,
    batch_counts: dict[str, int] | None = None,
    active_shard_ids_by_node: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    scale = int(resolved["session_scale"])
    batch_counts = batch_counts or {}
    active_by_node = active_shard_ids_by_node or {}
    node_records: dict[str, Any] = {}
    global_used_units = 0
    global_active_leases = 0
    global_active_executions = 0

    for node in sorted(spec_nodes, key=lambda item: item["node_id"]):
        node_id = node["node_id"]
        usage = active_usage.get(node_id, {"resource_units": 0, "leases": 0, "executions": 0})
        used_units = int(usage.get("resource_units", 0))
        lease_count = int(usage.get("leases", 0))
        execution_count = int(usage.get("executions", 0))
        global_used_units += used_units
        global_active_leases += lease_count
        global_active_executions += execution_count
        limits = effective_node_quota(node, resolved)
        cap_units = limits["resource_units_cap"]
        cap_q = q(cap_units, scale)
        used_q = q(used_units, scale)
        remaining_units = max(0, cap_units - used_units)
        remaining_q = max(0, cap_q - used_q)
        shard_error = sum(
            batch_accumulation_error_q(
                # Active usage map does not need per-shard units here; zero is safe and
                # scheduler candidate decisions add candidate-specific error separately.
                0,
                int(batch_counts.get(shard_id, 1)),
                scale,
            )
            for shard_id in active_by_node.get(node_id, [])
        )
        errors = quota_error_breakdown(
            scope=f"node:{node_id}",
            capacity_q=cap_q,
            scale=scale,
            resource_units=0,
            batch_count=1,
            baseline_scopes=1 if resolved.get("scale_changed_in_target") else 0,
        )
        errors["shard_accumulation_error_q"] = shard_error
        errors["synthetic_error_bound_q"] = (
            int(errors["quota_baseline_error_q"])
            + int(errors["shard_accumulation_error_q"])
            + int(errors["node_comparison_error_q"])
        )
        node_records[node_id] = {
            "node_id": node_id,
            "quota_version": resolved["version"],
            "session_scale": scale,
            "used_units": used_units,
            "resource_cap_units": cap_units,
            "remaining_units": remaining_units,
            "used_q": used_q,
            "capacity_q": cap_q,
            "remaining_q": remaining_q,
            "lease_limit": limits["lease_limit"],
            "active_leases": lease_count,
            "lease_limit_q": q(limits["lease_limit"], scale),
            "active_leases_q": q(lease_count, scale),
            "lease_remaining_q": q(max(0, limits["lease_limit"] - lease_count), scale),
            "concurrency_quota": limits["concurrency"],
            "active_executions": execution_count,
            "concurrency_q": q(limits["concurrency"], scale),
            "active_executions_q": q(execution_count, scale),
            "concurrency_remaining_q": q(max(0, limits["concurrency"] - execution_count), scale),
            "error_bounds_q": errors,
        }

    global_cap_units = int(resolved["global_budget_units"])
    global_cap_q = q(global_cap_units, scale)
    global_used_q = q(global_used_units, scale)
    global_remaining_q = max(0, global_cap_q - global_used_q)
    candidate_shard_error = sum(
        batch_accumulation_error_q(0, max(1, count), scale) for count in batch_counts.values()
    )
    global_errors = quota_error_breakdown(
        scope="global",
        capacity_q=global_cap_q,
        scale=scale,
        resource_units=0,
        batch_count=1,
        baseline_scopes=1 if resolved.get("scale_changed_in_target") else 0,
    )
    global_errors["shard_accumulation_error_q"] = candidate_shard_error
    global_errors["synthetic_error_bound_q"] = (
        int(global_errors["quota_baseline_error_q"])
        + int(global_errors["shard_accumulation_error_q"])
        + int(global_errors["node_comparison_error_q"])
    )
    global_record = {
        "scope": "global",
        "quota_version": resolved["version"],
        "session_scale": scale,
        "global_budget_units": global_cap_units,
        "used_units": global_used_units,
        "remaining_units": max(0, global_cap_units - global_used_units),
        "budget_q": global_cap_q,
        "used_q": global_used_q,
        "remaining_q": global_remaining_q,
        "active_leases": global_active_leases,
        "active_executions": global_active_executions,
        "error_bounds_q": global_errors,
        "node_water": node_records,
    }
    return global_record


def candidate_water_decision(
    shard: dict[str, Any],
    node: dict[str, Any],
    resolved: dict[str, Any],
    global_water: dict[str, Any],
    topology_order: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    scale = int(resolved["session_scale"])
    node_water = global_water["node_water"][node["node_id"]]
    demand_units = int(shard["resource_units"])
    demand_q = q(demand_units, scale)
    global_delta_q = int(global_water["remaining_q"]) - demand_q
    node_delta_q = int(node_water["remaining_q"]) - demand_q
    batch_count = int(shard.get("batch_count_hint", 1))
    global_bound = int(global_water["error_bounds_q"]["synthetic_error_bound_q"])
    node_bound = int(node_water["error_bounds_q"]["synthetic_error_bound_q"])
    candidate_shard_bound = batch_accumulation_error_q(demand_units, batch_count, scale)
    error_bound = global_bound + node_bound + candidate_shard_bound
    error_affected = abs(global_delta_q) <= error_bound or abs(node_delta_q) <= error_bound
    rank = {shard_id: index for index, shard_id in enumerate(topology_order)}
    return {
        "shard_id": shard["shard_id"],
        "node_id": node["node_id"],
        "topology_rank": rank.get(shard["shard_id"], 10**9),
        "tie_break_key": [rank.get(shard["shard_id"], 10**9), shard["shard_id"]],
        "demand_units": demand_units,
        "demand_q": demand_q,
        "global_difference_q": global_delta_q,
        "node_difference_q": node_delta_q,
        "error_bound_q": error_bound,
        "error_affected": error_affected,
        "tie_rule": "stable_topology_rank_then_shard_id",
    }


def choose_error_tie(candidate_ids: list[str], shards: list[dict[str, Any]]) -> str | None:
    if not candidate_ids:
        return None
    order = stable_topology(shards).order
    rank = {shard_id: index for index, shard_id in enumerate(order)}
    return sorted(candidate_ids, key=lambda shard_id: (rank.get(shard_id, 10**9), shard_id))[0]


def quota_change_scope(version_doc: dict[str, Any], parent_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    batch_keys = set(version_doc.get("upsert_batch_requirements", {})) | set(
        version_doc.get("delete_batch_requirements", [])
    )
    batch_shards = {key.split(":", 1)[-1] if ":" in key else key for key in batch_keys}
    global_value = version_doc.get("global_budget_units")
    scale_value = version_doc.get("session_scale")
    if parent_doc is None:
        global_changed = global_value is not None
        scale_changed = scale_value is not None
    else:
        global_changed = global_value is not None and global_value != parent_doc.get("global_budget_units")
        parent_scale = parent_doc.get("session_scale")
        if parent_scale is None:
            parent_scale = resolve_quota([parent_doc], parent_doc["version"])["session_scale"] if "version" in parent_doc else None
        scale_changed = scale_value is not None and scale_value != parent_scale
    return {
        "version": version_doc["version"],
        "node_ids": set(version_doc.get("upsert_node_quotas", {}))
        | set(version_doc.get("delete_node_quotas", [])),
        "batch_shard_ids": batch_shards,
        "global_budget_changed": global_changed,
        "scale_changed": scale_changed,
    }

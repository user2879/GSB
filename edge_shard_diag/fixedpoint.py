from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FP_MAX = 2**63 - 1
TRUNCATION = "floor_nonnegative_division"
SATURATION = f"clamp_to_{FP_MAX}"


def saturate(value: int) -> tuple[int, bool]:
    if value > FP_MAX:
        return FP_MAX, True
    if value < -FP_MAX:
        return -FP_MAX, True
    return int(value), False


def quantize_units(units: int, scale: int) -> dict[str, Any]:
    raw = int(units) * int(scale)
    q, saturated = saturate(raw)
    return {
        "raw_units": int(units),
        "scale": int(scale),
        "q": q,
        "truncation_error_q": 0,
        "trunc_direction": TRUNCATION,
        "overflow_saturation": SATURATION,
        "saturated": saturated,
    }


def fixed_ratio(numerator: int, denominator: int, scale: int) -> dict[str, Any]:
    """Return floor(numerator * scale / denominator) and its remainder."""

    numerator = int(numerator)
    denominator = int(denominator)
    scale = int(scale)
    if denominator <= 0:
        return {
            "q": FP_MAX,
            "remainder": 0,
            "numerator": numerator,
            "denominator": denominator,
            "scale": scale,
            "error_bound_q": FP_MAX,
            "trunc_direction": TRUNCATION,
            "overflow_saturation": SATURATION,
            "saturated": True,
            "zero_denominator": True,
        }
    raw = numerator * scale
    q = raw // denominator
    remainder = raw - q * denominator
    q, saturated = saturate(q)
    return {
        "q": q,
        "remainder": remainder,
        "numerator": numerator,
        "denominator": denominator,
        "scale": scale,
        "error_bound_q": 1,
        "error_bound_normalized": 1.0 / scale,
        "exact_fractional_error_q": remainder / denominator,
        "trunc_direction": TRUNCATION,
        "overflow_saturation": SATURATION,
        "saturated": saturated,
        "zero_denominator": False,
    }


def shard_error_record(shard_id: str, resource_units: int, budget_units: int, scale: int) -> dict[str, Any]:
    demand = quantize_units(resource_units, scale)
    level = fixed_ratio(resource_units, budget_units, scale)
    return {
        "shard_id": shard_id,
        "resource_units": resource_units,
        "demand_q": demand["q"],
        "demand_truncation_error_q": demand["truncation_error_q"],
        "budget_share_level_q": level["q"],
        "remainder": level["remainder"],
        "denominator": budget_units,
        "truncation_error_q_exact": f"{level['remainder']}/{budget_units}" if budget_units else "saturated",
        "absolute_error_bound_q": 1 if budget_units > 0 else FP_MAX,
        "normalized_error_bound": 1.0 / scale if budget_units > 0 else 1.0,
        "trunc_direction": TRUNCATION,
        "overflow_saturation": SATURATION,
    }


def node_water_record(
    node_id: str,
    used_units: int,
    quota_units: int,
    active_leases: int,
    lease_limit: int,
    scale: int,
    candidate_units: int = 0,
) -> dict[str, Any]:
    resource_level = fixed_ratio(used_units, quota_units, scale)
    projected_level = fixed_ratio(used_units + candidate_units, quota_units, scale)
    lease_level = fixed_ratio(active_leases, lease_limit, scale)
    return {
        "node_id": node_id,
        "used_units": used_units,
        "quota_units": quota_units,
        "used_q": quantize_units(used_units, scale)["q"],
        "quota_units_q": quantize_units(quota_units, scale)["q"],
        "active_leases": active_leases,
        "lease_limit": lease_limit,
        "active_leases_q": quantize_units(active_leases, scale)["q"],
        "lease_limit_q": quantize_units(lease_limit, scale)["q"],
        "candidate_units": candidate_units,
        "resource_level_q": resource_level["q"],
        "projected_resource_level_q": projected_level["q"],
        "lease_level_q": lease_level["q"],
        "comparison_error_bound_q": 2,
        "comparison_error_bound_normalized": 2.0 / scale,
        "trunc_direction": TRUNCATION,
        "overflow_saturation": SATURATION,
    }


def global_budget_record(used_units: int, budget_units: int, shard_count: int, scale: int) -> dict[str, Any]:
    remaining = max(0, budget_units - used_units)
    aggregate_level = fixed_ratio(used_units, budget_units, scale)
    return {
        "used_units": used_units,
        "budget_units": budget_units,
        "remaining_units": remaining,
        "used_q": quantize_units(used_units, scale)["q"],
        "budget_q": quantize_units(budget_units, scale)["q"],
        "remaining_q": quantize_units(remaining, scale)["q"],
        "aggregate_level_q": aggregate_level["q"],
        "aggregate_direct_error_bound_q": 1,
        "per_shard_sum_error_bound_q": max(1, shard_count),
        "budget_aggregation_error_upper_bound_q": max(1, shard_count),
        "budget_aggregation_error_normalized": max(1, shard_count) / scale,
        "trunc_direction": TRUNCATION,
        "overflow_saturation": SATURATION,
    }


def attribute_decision(
    *,
    candidate_ids: list[str],
    candidate_levels_q: list[int],
    slack_q: int,
    scale: int,
    capacity_changed_dimensions: int = 1,
    dependency_propagation_hops: int = 0,
    renewal_count: int = 0,
    unresolved_dependencies: int = 0,
) -> dict[str, Any]:
    """Build the explicit synthetic error envelope used for close decisions."""

    capacity_q = max(0, capacity_changed_dimensions)
    dependency_q = max(0, dependency_propagation_hops + unresolved_dependencies)
    lease_q = 1 + max(0, renewal_count)
    truncation_q = 2
    synthetic_bound_q = truncation_q + capacity_q + dependency_q + lease_q
    level_delta_q = abs(candidate_levels_q[0] - candidate_levels_q[1]) if len(candidate_levels_q) >= 2 else 0
    slack_close = abs(int(slack_q)) <= synthetic_bound_q
    level_close = len(candidate_levels_q) >= 2 and level_delta_q <= synthetic_bound_q
    affected = slack_close or level_close
    segments = {
        "capacity_generation_q": capacity_q,
        "dependency_readiness_q": dependency_q,
        "lease_renewal_q": lease_q,
        "quantization_comparison_q": truncation_q,
    }
    dominant = max(segments, key=lambda key: segments[key])
    return {
        "candidate_ids": candidate_ids,
        "candidate_levels_q": candidate_levels_q,
        "level_delta_q": level_delta_q,
        "slack_q": slack_q,
        "scale": scale,
        "segments_q": segments,
        "dominant_segment": dominant,
        "synthetic_error_bound_q": synthetic_bound_q,
        "synthetic_error_bound_normalized": synthetic_bound_q / scale,
        "threshold_rule": "error_affected iff abs(slack_q) <= bound or abs(level_delta_q) <= bound",
        "slack_close": slack_close,
        "level_close": level_close,
        "error_affected": affected,
        "tie_break_rule": "stable_topological_order_then_shard_id_ascending",
        "selected_id": candidate_ids[0] if candidate_ids else None,
    }

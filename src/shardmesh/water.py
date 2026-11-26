"""Fixed-point water marks and synthetic error-bound attribution."""

from __future__ import annotations

from typing import Any

from .fixedpoint import SATURATION, ceil_div, quantize_ratio, to_fixed_units
from .graph import all_upstreams


def dependency_depths(deps: dict[str, tuple[str, ...]] | dict[str, list[str]]) -> dict[str, int]:
    depths: dict[str, int] = {}

    def depth_of(shard_id: str) -> int:
        if shard_id in depths:
            return depths[shard_id]
        upstream = deps.get(shard_id, ())
        if not upstream:
            depths[shard_id] = 0
        else:
            depths[shard_id] = 1 + max(depth_of(dep) for dep in upstream)
        return depths[shard_id]

    for shard_id in deps:
        depth_of(shard_id)
    return depths


def marginal_quantization(used: int, added: int, capacity: int, scale: int) -> dict[str, Any]:
    before = quantize_ratio(used, capacity, scale)
    after = quantize_ratio(used + added, capacity, scale)
    marginal_q = after.q - before.q if not before.saturated and not after.saturated else SATURATION
    true_scaled_marginal_num = added * scale
    true_scaled_marginal_den = max(1, capacity)
    error_q_num = after.remainder - before.remainder
    return {
        "capacity": capacity,
        "used": used,
        "added": added,
        "before_q": before.q,
        "after_q": after.q,
        "marginal_q": marginal_q,
        "true_marginal_q_num": true_scaled_marginal_num,
        "true_marginal_q_den": true_scaled_marginal_den,
        "truncation_error_q_num": error_q_num,
        "truncation_error_q_den": max(1, capacity),
        "absolute_error_lt_q": 1,
        "resource_error_scaled_num": abs(error_q_num),
        "resource_error_scale": scale,
        "truncation": "floor_nonnegative_toward_zero",
    }


def build_attribution(
    *,
    shard_id: str,
    deps: dict[str, tuple[str, ...]] | dict[str, list[str]],
    scale: int,
    active_count: int,
    global_budget: int,
    node_quota: int,
    node_lease_limit: int,
    aggregation_count: int | None = None,
) -> dict[str, Any]:
    depths = dependency_depths(deps)
    depth = depths.get(shard_id, 0)
    ancestors = len(all_upstreams(shard_id, deps))
    single_q = 1
    propagation_q = depth
    direct_dependency_count = len(deps.get(shard_id, ()))
    aggregation_base = 1 + direct_dependency_count
    if aggregation_count is not None:
        aggregation_base += max(0, aggregation_count)
    aggregation_q = aggregation_base
    components = {
        "single_value_quantization_q": single_q,
        "dependency_propagation_q": propagation_q,
        "aggregation_accumulation_q": aggregation_q,
    }
    dominant = max(components, key=components.get)
    bound_q = single_q + propagation_q + aggregation_q
    return {
        "shard_id": shard_id,
        "scale": scale,
        "dependency_depth": depth,
        "ancestor_count": ancestors,
        "active_count": active_count,
        "direct_dependency_count": direct_dependency_count,
        "aggregation_count_used": aggregation_base,
        "components_q": components,
        "synthetic_bound_q": bound_q,
        "dominant_cause": dominant,
        "dominant_value_q": components[dominant],
        "decision_threshold": {
            "water_difference_q": bound_q,
            "global_budget_scaled_units": bound_q * max(1, global_budget),
            "node_concurrency_scaled_slots": bound_q * max(1, node_quota),
            "node_lease_scaled_slots": bound_q * max(1, node_lease_limit),
            "rule": "ambiguous_when_abs(delta)<=threshold; tie_break=(stable_topology_index,shard_id)",
        },
        "overflow_rule": f"saturate_at_{SATURATION}",
        "division_rule": "non_negative_integer_floor_truncation_toward_zero",
    }


def budget_boundary(
    *,
    used_units: int,
    candidate_units: int,
    budget: int,
    scale: int,
    bound_q: int,
) -> dict[str, Any]:
    remaining = budget - used_units
    slack = remaining - candidate_units
    slack_scaled = slack * scale
    threshold_scaled = bound_q * max(1, budget)
    ambiguous = abs(slack_scaled) <= threshold_scaled
    return {
        "remaining_units": remaining,
        "candidate_units": candidate_units,
        "slack_units": slack,
        "slack_scaled": slack_scaled,
        "error_bound_scaled_units": threshold_scaled,
        "ambiguous": ambiguous,
        "hard_admittable": slack >= 0,
        "explicit_threshold": f"abs({slack}*{scale}) <= {bound_q}*{max(1, budget)}",
    }


def node_water_view(
    *,
    active_shards: int,
    active_leases: int,
    concurrency: int,
    lease_limit: int,
    scale: int,
) -> dict[str, Any]:
    shard_water = quantize_ratio(active_shards, concurrency, scale)
    lease_water = quantize_ratio(active_leases, lease_limit, scale)
    after_shard_water = quantize_ratio(active_shards + 1, concurrency, scale)
    after_lease_water = quantize_ratio(active_leases + 1, lease_limit, scale)
    projected_q = max(after_shard_water.q, after_lease_water.q)
    return {
        "active_shards": active_shards,
        "active_leases": active_leases,
        "concurrency": concurrency,
        "lease_limit": lease_limit,
        "concurrency_water_q": shard_water.q,
        "lease_water_q": lease_water.q,
        "projected_after_admit_q": projected_q,
        "resource_fixed_multiplier": scale,
        "concurrency_limit_q": to_fixed_units(concurrency, scale),
        "lease_limit_fixed_q": to_fixed_units(lease_limit, scale),
        "zero_capacity_saturated": concurrency <= 0 or lease_limit <= 0,
    }


def water_boundary(left_q: int, right_q: int, left_bound_q: int, right_bound_q: int) -> dict[str, Any]:
    threshold = left_bound_q + right_bound_q
    delta = abs(left_q - right_q)
    return {
        "left_q": left_q,
        "right_q": right_q,
        "absolute_delta_q": delta,
        "synthetic_bound_q": threshold,
        "ambiguous": delta <= threshold,
        "tie_break": "(stable_topology_index, shard_id)",
    }


def global_aggregation_bound(per_shard_bounds_q: list[int], budget: int, scale: int = 1000) -> dict[str, Any]:
    accumulation_q = sum(max(0, value) for value in per_shard_bounds_q)
    direct_meter_q = 1
    conservative_q = direct_meter_q + accumulation_q
    return {
        "direct_meter_q": direct_meter_q,
        "sum_per_shard_bounds_q": accumulation_q,
        "global_upper_bound_q": conservative_q,
        "budget_equivalent_scaled_units": conservative_q * max(1, budget),
        "budget_equivalent_units_ceil": ceil_div(conservative_q * max(1, budget), max(1, scale)),
        "note": "budget_equivalent_scaled_units is in resource*scale fixed units",
    }

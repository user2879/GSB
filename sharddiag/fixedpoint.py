"""Integer fixed-point water levels and explicit error attribution."""

from __future__ import annotations

from typing import Any, Iterable


INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


def scale_for_power(power: int) -> int:
    if not isinstance(power, int) or isinstance(power, bool) or not 0 <= power <= 3:
        raise ValueError("session_scale must be an integer in [0, 3]")
    return 10**power


def saturate_int(value: int, *, lower: int = 0, upper: int = INT64_MAX) -> tuple[int, bool]:
    if value > upper:
        return upper, True
    if value < lower:
        return lower, True
    return int(value), False


def saturating_mul(left: int, right: int) -> tuple[int, bool]:
    value = left * right
    return saturate_int(value, lower=INT64_MIN, upper=INT64_MAX)


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-numerator) // denominator)


def floor_ratio_q(numerator: int, denominator: int, scale: int) -> dict[str, int | bool]:
    """Return floor(scale * numerator / denominator) and truncation residual.

    Non-negative ratios use floor/downward truncation.  The residual is the
    modulo remainder of ``scale * numerator`` divided by ``denominator``.
    """

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    scaled, saturated = saturating_mul(max(0, int(numerator)), scale)
    quotient, residual = divmod(scaled, denominator)
    quotient, quotient_saturated = saturate_int(quotient, lower=0, upper=scale)
    return {
        "water_q": int(quotient),
        "residual_num": int(residual),
        "residual_den": int(denominator),
        "error_bound_q_numerator": int(residual),
        "error_bound_q_denominator": int(denominator),
        "saturated": bool(saturated or quotient_saturated),
        "truncation_direction": "floor_down_for_nonnegative",
    }


def quantize_shard_demand(
    *, shard_id: str, resource_units: int, budget_units: int, scale_power: int
) -> dict[str, Any]:
    scale = scale_for_power(scale_power)
    scaled_units, units_saturated = saturating_mul(resource_units, scale)
    ratio = floor_ratio_q(resource_units, budget_units, scale)
    residual = int(ratio["residual_num"])
    denominator = int(ratio["residual_den"])
    return {
        "shard_id": shard_id,
        "resource_units": int(resource_units),
        "session_scale_power": int(scale_power),
        "session_scale": scale,
        "scaled_resource_q": scaled_units,
        "budget_pressure_q": ratio["water_q"],
        "truncation_error_q_exact": f"{residual}/{denominator}",
        "truncation_error_units_exact": f"{residual}/{scale}",
        "truncation_error_q_upper_exclusive": 1,
        "saturated": bool(units_saturated or ratio["saturated"]),
        "rounding": "floor_down",
    }


def quantize_node_water(
    *,
    node_id: str,
    used_units: int,
    active_shards: int,
    active_leases: int,
    global_budget_units: int,
    concurrency_quota: int,
    lease_limit: int,
    scale_power: int,
) -> dict[str, Any]:
    scale = scale_for_power(scale_power)
    resource = floor_ratio_q(used_units, global_budget_units, scale)
    used_scaled_q, used_saturated = saturating_mul(max(0, int(used_units)), scale)
    concurrency_limit_scaled_q, concurrency_limit_saturated = saturating_mul(
        max(0, int(concurrency_quota)), scale
    )
    lease_limit_scaled_q, lease_limit_saturated = saturating_mul(max(0, int(lease_limit)), scale)
    active_shards_scaled_q, active_shards_saturated = saturating_mul(
        max(0, int(active_shards)), scale
    )
    active_leases_scaled_q, active_leases_saturated = saturating_mul(
        max(0, int(active_leases)), scale
    )
    if concurrency_quota == 0:
        concurrency = {
            "water_q": scale if active_shards > 0 else 0,
            "blocked_by_zero_quota": True,
            "error_bound_q": 0 if active_shards == 0 else scale,
        }
    else:
        ratio = floor_ratio_q(active_shards, concurrency_quota, scale)
        concurrency = {**ratio, "blocked_by_zero_quota": False, "error_bound_q": 1}
    if lease_limit == 0:
        lease = {
            "water_q": scale if active_leases > 0 else 0,
            "blocked_by_zero_limit": True,
            "error_bound_q": 0 if active_leases == 0 else scale,
        }
    else:
        ratio = floor_ratio_q(active_leases, lease_limit, scale)
        lease = {**ratio, "blocked_by_zero_limit": False, "error_bound_q": 1}

    water_q = max(int(resource["water_q"]), int(concurrency["water_q"]), int(lease["water_q"]))
    return {
        "node_id": node_id,
        "used_units": int(used_units),
        "used_units_scaled_q": used_scaled_q,
        "active_shards": int(active_shards),
        "active_leases": int(active_leases),
        "concurrency_quota_scaled_q": concurrency_limit_scaled_q,
        "lease_limit_scaled_q": lease_limit_scaled_q,
        "active_shards_scaled_q": active_shards_scaled_q,
        "active_leases_scaled_q": active_leases_scaled_q,
        "resource_water_q": resource["water_q"],
        "concurrency_water_q": concurrency["water_q"],
        "lease_water_q": lease["water_q"],
        "node_water_q": water_q,
        "self_comparison_error_bound_q": 1,
        "pair_comparison_error_bound_q": 2,
        "scaled_integer_saturation": bool(
            used_saturated
            or concurrency_limit_saturated
            or lease_limit_saturated
            or active_shards_saturated
            or active_leases_saturated
        ),
        "resource_quantization": resource,
        "concurrency_quantization": concurrency,
        "lease_quantization": lease,
    }


def attribute_errors(
    *,
    candidate_count: int,
    budget_units: int,
    scale_power: int,
    lease_attribution_units: int = 0,
    dependency_wait_units: int = 0,
    extra_quantization_terms: int = 0,
) -> dict[str, Any]:
    """Build the three-part synthetic error bound.

    Quantization contributes fewer than one scaled water unit per floored
    term, so its conservative integer threshold is the term count.  Lease and
    dependency segments are upper bounds on resource pressure that can move
    into the decision while reconciliation/graph wait remains unresolved.
    """

    scale = scale_for_power(scale_power)
    quantization_bound_q = max(1, int(candidate_count) + int(extra_quantization_terms))
    lease_bound_q = ceil_div(scale * max(0, int(lease_attribution_units)), max(1, int(budget_units)))
    dependency_bound_q = ceil_div(
        scale * max(0, int(dependency_wait_units)), max(1, int(budget_units))
    )
    synthetic_bound_q = quantization_bound_q + lease_bound_q + dependency_bound_q
    threshold_units = ceil_div(synthetic_bound_q * max(1, int(budget_units)), scale)

    segments = {
        "quantization_truncation": quantization_bound_q,
        "lease_attribution": lease_bound_q,
        "dependency_wait": dependency_bound_q,
    }
    priority = {
        "dependency_wait": 3,
        "lease_attribution": 2,
        "quantization_truncation": 1,
    }
    dominant_segment = max(segments, key=lambda name: (segments[name], priority[name]))
    return {
        "scale": scale,
        "synthetic_error_bound_q": synthetic_bound_q,
        "decision_threshold_resource_units": threshold_units,
        "segments_q": segments,
        "dominant_segment": dominant_segment,
        "dominant_bound_q": segments[dominant_segment],
        "overflow_saturation_rule": "clamp_to_signed_int64_and_mark_saturated",
        "truncation_rule": "floor_down_for_nonnegative_and_toward_negative_infinity_for_signed",
    }


def budget_margin_decision(
    *, used_units: int, candidate_units: int, budget_units: int, attribution: dict[str, Any]
) -> dict[str, Any]:
    margin = int(budget_units) - int(used_units) - int(candidate_units)
    bound_q = int(attribution["synthetic_error_bound_q"])
    scale = int(attribution["scale"])
    within_error = abs(scale * margin) <= bound_q * int(budget_units)
    return {
        "budget_margin_units": margin,
        "exact_feasible": margin >= 0,
        "within_synthetic_error_bound": within_error,
        "decision_affected_by_error": within_error,
        "tie_break": "stable_topological_then_shard_id",
    }


def water_gap_decision(
    *, water_a_q: int, water_b_q: int, attribution: dict[str, Any]
) -> dict[str, Any]:
    gap = abs(int(water_a_q) - int(water_b_q))
    bound = int(attribution["synthetic_error_bound_q"])
    return {
        "water_gap_q": gap,
        "synthetic_error_bound_q": bound,
        "within_synthetic_error_bound": gap <= bound,
        "decision_affected_by_error": gap <= bound,
        "tie_break": "stable_topological_then_shard_id",
    }


def global_budget_error(per_shard: Iterable[dict[str, Any]], *, shard_count: int, scale_power: int) -> dict[str, Any]:
    scale = scale_for_power(scale_power)
    count = int(shard_count)
    return {
        "aggregated_shard_count": count,
        "pressure_error_upper_bound_q_exclusive": count,
        "normalized_error_upper_bound_exclusive": f"{count}/{scale}",
        "safe_integer_threshold_q": count,
        "note": "Direct global floor has error < 1_q; sum of per-shard floors has error < n_q.",
    }

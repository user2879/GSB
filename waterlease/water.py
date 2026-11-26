from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable


INT64_MAX = 2**63 - 1
ATTRIBUTION_RESOLUTION = 1_000_000_000
DOMINANCE_THRESHOLD_BP = 5000


class FixedPointMeter:
    """Integer fixed-point water-level arithmetic.

    ``scale_exp`` is 0..3 and ``scale = 10 ** scale_exp``. Rational values are
    quantized as ``floor(numerator * scale / denominator)`` for non-negative
    quantities, i.e. truncation toward zero/floor. If a q-value exceeds signed
    int64 range it is saturated to ``INT64_MAX`` rather than wrapped.
    """

    def __init__(self, scale_exp: int) -> None:
        if not isinstance(scale_exp, int) or isinstance(scale_exp, bool) or not 0 <= scale_exp <= 3:
            raise ValueError("scale_exp must be an integer in [0, 3]")
        self.scale_exp = scale_exp
        self.scale = 10**scale_exp

    def quantize_integer(self, value: int) -> dict[str, Any]:
        if value < 0:
            raise ValueError("water quantities must be non-negative")
        raw_q = value * self.scale
        saturated = raw_q > INT64_MAX
        return {
            "raw_units": value,
            "quantized_q": INT64_MAX if saturated else raw_q,
            "scale": self.scale,
            "truncation_error": 0,
            "saturated": saturated,
            "truncation_direction": "floor_nonnegative",
            "overflow_rule": "saturate_to_signed_int64_max",
        }

    def quantize_ratio(self, numerator: int, denominator: int, *, name: str) -> dict[str, Any]:
        if denominator <= 0:
            raise ValueError("denominator must be positive")
        if numerator < 0:
            raise ValueError("numerator must be non-negative")
        scaled = numerator * self.scale
        q = scaled // denominator
        remainder = scaled - q * denominator
        saturated = q > INT64_MAX
        safe_q = INT64_MAX if saturated else q
        error_num = remainder
        error_den = denominator * self.scale
        exact_error = Fraction(error_num, error_den) if not saturated else None
        units = (
            error_num * ATTRIBUTION_RESOLUTION // error_den
            if not saturated
            else ATTRIBUTION_RESOLUTION
        )
        return {
            "name": name,
            "scale_exp": self.scale_exp,
            "scale": self.scale,
            "numerator": numerator,
            "denominator": denominator,
            "quantized_q": safe_q,
            "remainder": remainder,
            "truncation_direction": "floor_nonnegative",
            "truncation_error_num": error_num,
            "truncation_error_den": error_den,
            "truncation_error_ppm_floor": (error_num * 1_000_000) // error_den,
            "attribution_units_1e9": units,
            "exact_error": str(exact_error) if exact_error is not None else "saturated",
            "saturated": saturated,
            "overflow_rule": "saturate_to_signed_int64_max",
        }


def attribute_shard(
    meter: FixedPointMeter,
    *,
    shard_id: str,
    topo_rank: int,
    total_shards: int,
    resource_units: int,
    capacity_budget_units: int,
    node_max_leases: int,
) -> dict[str, Any]:
    capacity = meter.quantize_ratio(
        resource_units,
        capacity_budget_units,
        name="capacity_version_chain",
    )
    if node_max_leases <= 0:
        lease = {
            "name": "lease_generation",
            "quantized_q": 0,
            "truncation_error_num": 0,
            "truncation_error_den": 1,
            "truncation_error_ppm_floor": 0,
            "attribution_units_1e9": 0,
            "saturated": False,
            "denominator": 0,
        }
    else:
        lease = meter.quantize_ratio(1, node_max_leases, name="lease_generation")
    queue = meter.quantize_ratio(
        topo_rank + 1,
        max(1, total_shards),
        name="schedule_queue",
    )
    segments_list = [capacity, lease, queue]
    total_units = sum(int(item["attribution_units_1e9"]) for item in segments_list)
    exact_total = sum(
        (
            Fraction(int(item["truncation_error_num"]), int(item["truncation_error_den"]))
            for item in segments_list
        ),
        Fraction(0, 1),
    )
    shares: dict[str, int] = {}
    for segment in segments_list:
        name = str(segment["name"])
        units = int(segment["attribution_units_1e9"])
        shares[name] = units * 10_000 // total_units if total_units else 0
    dominant = None
    for name, basis_points in shares.items():
        if basis_points > DOMINANCE_THRESHOLD_BP:
            dominant = name
            break

    resource_q = meter.quantize_integer(resource_units)
    return {
        "shard_id": shard_id,
        "scale_exp": meter.scale_exp,
        "scale": meter.scale,
        "resource_units": resource_units,
        "resource_units_q": resource_q["quantized_q"],
        "resource_amount_truncation_error": 0,
        "budget_share_q": capacity["quantized_q"],
        "quantization_truncation_error_num": exact_total.numerator,
        "quantization_truncation_error_den": exact_total.denominator,
        "quantization_truncation_error_ppm_floor": (
            exact_total.numerator * 1_000_000
        )
        // exact_total.denominator,
        "attribution_units_1e9": total_units,
        "attribution_basis_points": shares,
        "dominant_segment": dominant,
        "dominance_threshold_bp": DOMINANCE_THRESHOLD_BP,
        "dominance_rule": "segment is dominant only when share is strictly > 5000 bp",
        "segments": {item["name"]: item for item in segments_list},
    }


def summarize_attribution(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    segment_names = ("capacity_version_chain", "lease_generation", "schedule_queue")
    totals = {name: 0 for name in segment_names}
    for record in items:
        for name in segment_names:
            totals[name] += int(
                record.get("segments", {}).get(name, {}).get("attribution_units_1e9", 0)
            )
    grand_total = sum(totals.values())
    shares = {
        name: (value * 10_000 // grand_total if grand_total else 0)
        for name, value in totals.items()
    }
    dominant = None
    for name, basis_points in sorted(shares.items(), key=lambda pair: (-pair[1], pair[0])):
        if basis_points > DOMINANCE_THRESHOLD_BP:
            dominant = name
            break
    scale_exp = int(items[0]["scale_exp"]) if items else 0
    scale = 10**scale_exp
    shard_count = len(items)
    return {
        "shard_count": shard_count,
        "scale_exp": scale_exp,
        "scale": scale,
        "segment_attribution_units_1e9": totals,
        "segment_share_basis_points": shares,
        "segment_share_percent_floor": {name: bp // 100 for name, bp in shares.items()},
        "dominant_segment": dominant,
        "dominance_threshold_bp": DOMINANCE_THRESHOLD_BP,
        "dominance_rule": "strictly greater than 50.00% (5000 basis points)",
        "global_budget_aggregation_error_upper_bound": {
            "bound_num": shard_count,
            "bound_den": scale,
            "strict_relation": "error < bound",
            "fixed_q_units_bound": shard_count,
            "ppm_floor": shard_count * 1_000_000 // scale,
        },
    }


def node_water_error_bounds(
    *,
    node_id: str,
    shard_count: int,
    scale_exp: int,
    concurrency_quota: int,
    max_leases: int,
    used_active_shards: int | None = None,
    used_active_leases: int | None = None,
) -> dict[str, Any]:
    scale = 10**scale_exp
    result = {
        "node_id": node_id,
        "scale": scale,
        "concurrency_quota_units": concurrency_quota,
        "concurrency_quota_q": concurrency_quota * scale,
        "lease_cap_units": max_leases,
        "lease_cap_q": max_leases * scale,
        "direct_water_comparison_error_bound_num": 1,
        "direct_water_comparison_error_bound_den": scale,
        "direct_fixed_q_error_bound": 1,
        "summed_shard_error_bound_num": max(1, shard_count),
        "summed_shard_error_bound_den": scale,
        "safe_comparison_margin_q": max(1, shard_count),
        "strict_relation": "direct error < 1/scale; summed per-shard error < shard_count/scale",
        "concurrency_quota": concurrency_quota,
        "max_leases": max_leases,
        "zero_capacity_note": "zero quota or lease cap makes matching active shards unschedulable",
    }
    if used_active_shards is not None:
        result["used_active_shards"] = used_active_shards
        result["used_active_shards_q"] = used_active_shards * scale
    if used_active_leases is not None:
        result["used_active_leases"] = used_active_leases
        result["used_active_leases_q"] = used_active_leases * scale
    return result

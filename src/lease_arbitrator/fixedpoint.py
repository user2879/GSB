from __future__ import annotations

from typing import Any

import numpy as np


DOMINANT_SEGMENT_THRESHOLD = 0.60
INT64_MAX = (1 << 63) - 1


class FixedScale:
    def __init__(self, exponent: int):
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise ValueError("scale exponent must be an integer")
        if exponent not in (0, 1, 2, 3):
            raise ValueError("scale exponent must be in [0, 3]")
        self.exponent = exponent
        self.factor = 10**exponent


def make_scale(exponent: int) -> FixedScale:
    return FixedScale(exponent)


def saturate_nonnegative(value: int, cap: int = INT64_MAX) -> tuple[int, bool]:
    if value < 0:
        raise ValueError("fixed-point arithmetic is defined for nonnegative values")
    if value > cap:
        return cap, True
    return value, False


def quantize_units(units: int, scale: FixedScale) -> dict[str, Any]:
    value, saturated = saturate_nonnegative(units * scale.factor)
    return {
        "q": value,
        "scale_exponent": scale.exponent,
        "factor": scale.factor,
        "direct_truncation_error_units": 0,
        "saturated": saturated,
    }


def water_level_fp(used: int, quota: int, scale: FixedScale) -> dict[str, Any]:
    saturated = False
    if quota <= 0:
        level = scale.factor
        remainder_num = 0
        denominator = 1
        saturated = True
    else:
        level = (used * scale.factor) // quota
        remainder_num = (used * scale.factor) % quota
        denominator = quota
    level, level_saturated = saturate_nonnegative(level)
    saturated = saturated or level_saturated
    return {
        "level_fp": level,
        "factor": scale.factor,
        "used_q": min(used * scale.factor, INT64_MAX),
        "quota_q": min(quota * scale.factor, INT64_MAX),
        "remaining_q": min(max(quota - used, 0) * scale.factor, INT64_MAX),
        "truncation_remainder_numerator": remainder_num,
        "truncation_remainder_denominator": denominator,
        "ratio_ulp_error_bound": "less_than_1_level_fp",
        "saturated": saturated,
        "zero_quota_convention": "level_fp=factor/full-no-capacity" if quota <= 0 else None,
    }


def lease_quantization_error(units: int, quota: int, scale: FixedScale) -> dict[str, Any]:
    base = quantize_units(units, scale)
    effective_quota = quota if quota > 0 else units
    level = water_level_fp(min(units, effective_quota), effective_quota, scale)
    return {
        "lease_units": units,
        "direct_quantized_q": base["q"],
        "single_lease_quantization_error_units": 0,
        "derived_water_level_fp": level["level_fp"],
        "derived_truncation_residual": level["truncation_remainder_numerator"],
        "ratio_error_bound_fp": 1,
        "equivalent_units_error_bound_exact": f"{quota}/{scale.factor}",
        "equivalent_units_error_bound_decimal": quota / scale.factor if quota else 0.0,
    }


def node_comparison_bound_fp() -> int:
    return 2


def global_aggregation_bound_fp(node_count: int) -> int:
    return node_count + 1


def attribute_error_segments(
    *,
    quantization_ulp: int,
    scheduling_order_ulp: int,
    generation_reorder_ulp: int,
    scale: FixedScale,
) -> dict[str, Any]:
    segments = {
        "quantization_truncation": max(0, int(quantization_ulp)),
        "scheduling_order": max(0, int(scheduling_order_ulp)),
        "generation_reorder": max(0, int(generation_reorder_ulp)),
    }
    values = np.array(list(segments.values()), dtype=np.float64)
    total = int(values.sum())
    if total <= 0:
        segments["quantization_truncation"] = 1
        values = np.array([1.0, 0.0, 0.0])
        total = 1
    shares = values / total
    dominant_index = int(np.argmax(shares))
    dominant_name = list(segments)[dominant_index]
    dominant_share = float(shares[dominant_index])
    declared = dominant_name if dominant_share >= DOMINANT_SEGMENT_THRESHOLD else None
    return {
        "segments_ulp": segments,
        "segment_shares": {
            name: round(float(share), 6) for name, share in zip(segments, shares, strict=True)
        },
        "total_bound_ulp": total,
        "bound_units_exact": f"{total}/{scale.factor}",
        "bound_units_decimal": total / scale.factor,
        "dominant_threshold": DOMINANT_SEGMENT_THRESHOLD,
        "attribution": "dominant" if declared else "mixed_below_threshold",
        "dominant_segment": declared,
        "rule": "dominant only when numpy share is >= 0.60; otherwise mixed",
    }


def within_integer_bound(value_fp: int, bound_fp: int) -> bool:
    return abs(int(value_fp)) <= int(bound_fp)


def saturation_rule() -> dict[str, Any]:
    return {
        "integer_format": "signed-64-bit fixed-point compatible",
        "cap": INT64_MAX,
        "overflow": "saturate to cap and mark saturated=true",
        "truncation": "floor division for nonnegative quantities",
        "multiplication_division": "integer arithmetic only in capacity and boundary decisions",
    }

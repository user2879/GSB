"""Explicit integer fixed-point watermarks and error accounting.

Policy
------
* Domain quantities are non-negative integer resource units.
* Unit conversion is exact multiplication by ``scale`` unless it would exceed
  signed 64-bit ``MAX_I64``; overflow saturates at ``MAX_I64``.
* Ratios use floor division (truncation toward zero for non-negative values).
* Every truncated ratio contributes less than one fixed-point unit (``1`` in
  the returned denominator ``scale``).
* No comparison in this module converts to float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_I64 = (1 << 63) - 1
TRUNCATION = "floor_non_negative"
OVERFLOW = "saturate_i64"


def saturating_mul(value: int, factor: int) -> int:
    if value < 0 or factor < 0:
        raise ValueError("fixed-point helpers accept non-negative integers only")
    if value != 0 and factor > MAX_I64 // value:
        return MAX_I64
    return min(MAX_I64, value * factor)


def units_to_fixed(units: int, scale: int) -> int:
    if units < 0 or scale <= 0:
        raise ValueError("units must be non-negative and scale must be positive")
    return saturating_mul(units, scale)


def ratio_fixed(numerator_units: int, denominator_units: int, scale: int) -> tuple[int, int, bool]:
    """Return ``(floor(numerator/denominator*scale), remainder, saturated)``.

    Python integers safely hold the intermediate product.  Only a quotient
    that cannot fit signed 64-bit is saturated, which preserves the ratio's
    truncation semantics for all representable results.
    """

    if numerator_units < 0 or denominator_units <= 0 or scale <= 0:
        raise ValueError("invalid fixed-point ratio operands")
    scaled = numerator_units * scale
    quotient = scaled // denominator_units
    remainder = scaled % denominator_units
    saturated = quotient > MAX_I64
    return (MAX_I64 if saturated else quotient), remainder, saturated


@dataclass(frozen=True)
class WaterLevel:
    node_id: str
    scale: int
    used_units: int
    quota_units: int
    quota_fp: int
    used_fp: int
    lease_limit_units: int
    lease_limit_fp: int
    utilization_fp: int
    utilization_remainder: int
    headroom_fp: int
    headroom_remainder: int
    margin_units: int
    unit_overflowed: bool
    ratio_overflowed: bool
    overflowed: bool
    per_ratio_error_bound_fp: int = 1
    truncation: str = TRUNCATION
    overflow_rule: str = OVERFLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "scale": self.scale,
            "used_units": self.used_units,
            "quota_units": self.quota_units,
            "quota_fp": self.quota_fp,
            "used_fp": self.used_fp,
            "lease_limit_units": self.lease_limit_units,
            "lease_limit_fp": self.lease_limit_fp,
            "utilization_fp": self.utilization_fp,
            "utilization_remainder": self.utilization_remainder,
            "headroom_fp": self.headroom_fp,
            "headroom_remainder": self.headroom_remainder,
            "margin_units": self.margin_units,
            "unit_overflowed": self.unit_overflowed,
            "ratio_overflowed": self.ratio_overflowed,
            "overflowed": self.overflowed,
            "per_ratio_error_bound_fp": self.per_ratio_error_bound_fp,
            "truncation": self.truncation,
            "overflow_rule": self.overflow_rule,
        }


def water_level(
    node_id: str,
    used_units: int,
    quota_units: int,
    scale: int,
    *,
    lease_limit_units: int = 0,
) -> WaterLevel:
    if used_units < 0 or quota_units <= 0:
        raise ValueError("used_units must be non-negative and quota_units positive")
    if lease_limit_units < 0:
        raise ValueError("lease_limit_units must be non-negative")
    margin = max(0, quota_units - used_units)
    utilization, util_remainder, util_overflow = ratio_fixed(used_units, quota_units, scale)
    headroom, head_remainder, head_overflow = ratio_fixed(margin, quota_units, scale)
    quota_fp = units_to_fixed(quota_units, scale)
    used_fp = units_to_fixed(used_units, scale)
    lease_limit_fp = units_to_fixed(lease_limit_units, scale)
    unit_overflowed = any(
        value > MAX_I64 // scale
        for value in (used_units, quota_units, lease_limit_units)
    )
    ratio_overflowed = util_overflow or head_overflow
    return WaterLevel(
        node_id=node_id,
        scale=scale,
        used_units=used_units,
        quota_units=quota_units,
        quota_fp=quota_fp,
        used_fp=used_fp,
        lease_limit_units=lease_limit_units,
        lease_limit_fp=lease_limit_fp,
        utilization_fp=utilization,
        utilization_remainder=util_remainder,
        headroom_fp=headroom,
        headroom_remainder=head_remainder,
        margin_units=margin,
        unit_overflowed=unit_overflowed,
        ratio_overflowed=ratio_overflowed,
        overflowed=unit_overflowed or ratio_overflowed,
    )


@dataclass(frozen=True)
class FixedErrorModel:
    scale: int
    node_count: int
    per_shard_quantization_fp: int
    per_node_quantization_fp: int
    per_node_comparison_bound_fp: int
    global_aggregation_bound_fp: int
    quantization_contribution_fp: int
    comparison_contribution_fp: int
    aggregation_contribution_fp: int
    total_bound_fp: int
    dominant_segment: str
    tie_break_rule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "node_count": self.node_count,
            "per_shard_quantization_fp": self.per_shard_quantization_fp,
            "per_node_quantization_fp": self.per_node_quantization_fp,
            "per_node_comparison_bound_fp": self.per_node_comparison_bound_fp,
            "global_aggregation_bound_fp": self.global_aggregation_bound_fp,
            "components_fp": {
                "quantization_truncation": self.quantization_contribution_fp,
                "comparison_decision": self.comparison_contribution_fp,
                "aggregation_accumulation": self.aggregation_contribution_fp,
            },
            "total_bound_fp": self.total_bound_fp,
            "dominant_segment": self.dominant_segment,
            "tie_break_rule": self.tie_break_rule,
            "truncation": TRUNCATION,
            "overflow_rule": OVERFLOW,
            "decision_source": "integer_fixed_point_no_float",
        }


def analyze_fixed_error(node_count: int, scale: int) -> FixedErrorModel:
    if node_count < 1 or scale <= 0:
        raise ValueError("node_count and scale must be positive")
    per_node_quant = 2  # utilization + headroom
    comparison = 2  # one utilization truncation per compared candidate
    aggregation = node_count  # summing node watermarks
    quantization_total = per_node_quant * node_count
    components = {
        "quantization_truncation": quantization_total,
        "comparison_decision": comparison,
        "aggregation_accumulation": aggregation,
    }
    priority = {"quantization_truncation": 2, "comparison_decision": 1, "aggregation_accumulation": 0}
    dominant = max(components, key=lambda key: (components[key], priority[key]))
    return FixedErrorModel(
        scale=scale,
        node_count=node_count,
        per_shard_quantization_fp=1,
        per_node_quantization_fp=per_node_quant,
        per_node_comparison_bound_fp=comparison,
        global_aggregation_bound_fp=aggregation,
        quantization_contribution_fp=quantization_total,
        comparison_contribution_fp=comparison,
        aggregation_contribution_fp=aggregation,
        total_bound_fp=sum(components.values()),
        dominant_segment=dominant,
        tie_break_rule="quantization_truncation > comparison_decision > aggregation_accumulation",
    )

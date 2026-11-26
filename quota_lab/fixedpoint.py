from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

ROUNDING_RULE = "floor_toward_negative_infinity"
SATURATION_RULE = "saturate_to_signed_int64_bounds"


@dataclass(frozen=True)
class DivisionResult:
    quotient: int
    remainder: int
    denominator: int
    bound_q: int
    saturated: bool = False

    @property
    def residual(self) -> dict[str, int]:
        return {"numerator": self.remainder, "denominator": self.denominator}


def saturation_info(value: int) -> tuple[int, bool]:
    if value > INT64_MAX:
        return INT64_MAX, True
    if value < INT64_MIN:
        return INT64_MIN, True
    return value, False


def quantize(resource_units: int, q: int) -> int:
    value = resource_units * q
    saturated, _ = saturation_info(value)
    return saturated


def floor_divide(numerator: int, denominator: int) -> DivisionResult:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient = numerator // denominator
    remainder = numerator - quotient * denominator
    quotient, saturated = saturation_info(quotient)
    if remainder == 0:
        bound = 0
    else:
        bound = 1 if remainder > 0 else 1
    return DivisionResult(
        quotient=quotient,
        remainder=remainder,
        denominator=denominator,
        bound_q=bound,
        saturated=saturated,
    )


def error_vector(
    *,
    shard_bound_q: int,
    node_bound_q: int,
    global_bound_q: int,
    shard_tolerance_q: int = 0,
    node_tolerance_q: int = 0,
    global_tolerance_q: int = 0,
) -> dict[str, Any]:
    shard_component = shard_bound_q + shard_tolerance_q
    node_component = node_bound_q + node_tolerance_q
    global_component = global_bound_q + global_tolerance_q
    composite = shard_component + node_component + global_component
    return {
        "components_q": {
            "shard": shard_component,
            "node": node_component,
            "global": global_component,
        },
        "raw_bounds_q": {
            "shard_truncation": shard_bound_q,
            "node_rounding": node_bound_q,
            "global_rounding": global_bound_q,
        },
        "tolerances_q": {
            "shard": shard_tolerance_q,
            "node_watermark": node_tolerance_q,
            "global_budget": global_tolerance_q,
        },
        "composite_bound_q": composite,
        "rounding_rule": ROUNDING_RULE,
        "saturation_rule": SATURATION_RULE,
    }


def contribution_report(
    shard_contribution_q: int,
    node_contribution_q: int,
    global_contribution_q: int,
    *,
    dominance_threshold_bp: int,
) -> dict[str, Any]:
    values = {
        "shard": max(0, shard_contribution_q),
        "node": max(0, node_contribution_q),
        "global": max(0, global_contribution_q),
    }
    total = sum(values.values())
    if total <= 0:
        shares = {"shard": 0, "node": 0, "global": 0}
        remainder_bp = 10000
        dominant = "none_zero_error"
    else:
        shares = {level: value * 10000 // total for level, value in values.items()}
        remainder_bp = 10000 - sum(shares.values())
        largest_level = max(("shard", "node", "global"), key=lambda level: (values[level], level))
        largest_share = values[largest_level] * 10000 // total
        if largest_share >= dominance_threshold_bp:
            dominant = largest_level
        else:
            dominant = "mixed_below_threshold"
    return {
        "contributions_q": values,
        "total_q": total,
        "share_basis_points_floor": shares,
        "share_rounding_remainder_bp": remainder_bp,
        "dominance_threshold_bp": dominance_threshold_bp,
        "dominant_level": dominant,
        "integer_rule": "share_bp = contribution_q * 10000 // total_q",
    }


def within_integer_bound(difference_q: int, bound_q: int) -> bool:
    return abs(difference_q) <= max(0, bound_q)

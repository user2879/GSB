"""Integer fixed-point watermark diagnostics with explicit error bounds.

Ratios are stored in a session-scaled fixed-point scale (default 10_000, i.e.
basis points with two decimals).  Truncation is always toward zero (floor for
non-negative values).  A single value carries an absolute quantization error
of one ulp; comparing two ratios therefore uses a conservative combined
error bound of two ulps.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .model import Pool

TRUNC_DIRECTION = "toward_zero"
DEFAULT_SCALE = 10_000


def fixed_ratio(numerator: int, denominator: int,
                scale: int = DEFAULT_SCALE) -> int:
    """Return ``numerator / denominator * scale`` truncated toward zero."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) * int(scale)) // int(denominator)


@dataclass
class PoolWatermark:
    pool_id: str
    effective_leases: int
    capacity: int
    used_capacity: int
    queue_length: int
    ratio_fp: int
    scale: int
    truncation: str
    value_error_ulp: int
    value_error_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GlobalWatermark:
    effective_leases: int
    capacity: int
    used_capacity: int
    queue_length: int
    ratio_fp: int
    scale: int
    truncation: str
    value_error_ulp: int
    value_error_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def pool_watermark(pool: Pool, effective_leases: int, queue_length: int,
                   scale: int = DEFAULT_SCALE) -> PoolWatermark:
    ratio = fixed_ratio(effective_leases, pool.capacity, scale)
    return PoolWatermark(
        pool.pool_id, effective_leases, pool.capacity, effective_leases,
        queue_length, ratio, scale, TRUNC_DIRECTION, 1, 1.0 / scale)


def global_watermark(pools: dict[str, Pool], effective_by_pool: dict[str, int],
                     queue_by_pool: dict[str, int],
                     scale: int = DEFAULT_SCALE) -> GlobalWatermark:
    total_cap = sum(pool.capacity for pool in pools.values())
    total_used = sum(effective_by_pool.get(pid, 0) for pid in pools)
    total_queue = sum(queue_by_pool.get(pid, 0) for pid in pools)
    ratio = fixed_ratio(total_used, total_cap, scale)
    return GlobalWatermark(
        total_used, total_cap, total_used, total_queue, ratio, scale,
        TRUNC_DIRECTION, 1, 1.0 / scale)


def comparison_error_bound(scale: int = DEFAULT_SCALE) -> dict:
    """Combined bound when two fixed-point values are compared."""
    return {"comparison_error_ulp": 2,
            "comparison_error_ratio": 2.0 / scale,
            "scale": scale, "truncation": TRUNC_DIRECTION}


def rank_pools(watermarks: list[PoolWatermark],
               scale: int = DEFAULT_SCALE) -> list[dict]:
    """Stable ranking with error-influenced tie annotation.

    Primary order is descending fixed-point ratio.  When the difference is no
    more than the combined 2-ulp error bound the order is decided by a stable
    tie-break (higher capacity, then pool_id) and explicitly flagged.
    """
    ordered = sorted(
        watermarks,
        key=lambda item: (-item.ratio_fp, -item.capacity, item.pool_id))
    bound = comparison_error_bound(scale)
    rows: list[dict] = []
    for position, item in enumerate(ordered):
        within_error = False
        if position > 0:
            previous = ordered[position - 1]
            within_error = (previous.ratio_fp - item.ratio_fp) <= 2
        rows.append({
            "rank": position + 1,
            "pool_id": item.pool_id,
            "ratio_fp": item.ratio_fp,
            "difference_from_previous_ulp":
                None if position == 0
                else ordered[position - 1].ratio_fp - item.ratio_fp,
            "within_combined_error_bound": within_error,
            "error_influenced": within_error,
            "bound": bound,
        })
    return rows

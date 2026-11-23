"""Integer fixed-point statistics with explicit error bounds.

Session scale factor is an integer power of ten: ``S = 10**k``. Every ratio or
mean is stored as a non-negative integer fixed-point value using explicit
*floor* truncation: ``fixed = floor(numerator * S / denominator)``. All
intermediate sums are exact integers; exactly one division (hence one floor
truncation) is performed per statistic.

Error bounds are reported in ULP units (1 ULP = 1/S) as integer counts, split
into three documented contributions:
    quant   - quantization / floor truncation (at most 1 ULP)
    window  - silence-window closed-interval boundary attribution ambiguity
    layer   - dependency max-layer cutoff normalization ambiguity
The largest non-zero contribution is flagged as the dominant segment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import AggregateAlert, Correlation

QUANT = "quantization_truncation"
WINDOW = "silence_window_boundary"
LAYER = "dependency_layer_normalization"


def floor_fixed(numerator: int, denominator: int, scale: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) * int(scale)) // int(denominator)


def _ulp(real_bound: float, scale: int) -> int:
    # Conservative integer ULP count for a real-valued bound.
    return max(1, math.ceil(real_bound * scale - 1e-12)) if real_bound > 0 else 0


@dataclass(frozen=True)
class FixedStat:
    metric: str
    key: str
    numerator: int
    denominator: int
    scale_k: int
    fixed_value: int
    truncation: str
    ulp_breakdown: dict[str, int]
    dominant_segment: str
    error_indistinguishable: bool = False

    @property
    def scale(self) -> int:
        return 10 ** self.scale_k

    @property
    def total_ulp(self) -> int:
        return sum(self.ulp_breakdown.values())

    def decimal(self) -> str:
        s = self.scale
        return f"{self.fixed_value // s}.{self.fixed_value % s:0{self.scale_k}d}"

    def to_record(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "key": self.key,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "scale_factor": 10 ** self.scale_k,
            "truncation": self.truncation,
            "fixed_value": self.fixed_value,
            "decimal": self.decimal(),
            "ulp_error_breakdown": dict(self.ulp_breakdown),
            "total_error_ulp": self.total_ulp,
            "dominant_segment": self.dominant_segment,
            "error_indistinguishable": self.error_indistinguishable,
        }


def _dominant(breakdown: dict[str, int]) -> str:
    nonzero = {k: v for k, v in breakdown.items() if v > 0}
    if not nonzero:
        return QUANT
    return max(sorted(nonzero), key=lambda k: nonzero[k])


def suppression_ratios(aggregates: list[AggregateAlert], scale_k: int = 4
                       ) -> list[FixedStat]:
    """Per-fingerprint (plus global) suppressed/total fixed-point ratios."""
    scale = 10 ** scale_k
    by_fp: dict[str, list[AggregateAlert]] = {}
    for agg in aggregates:
        by_fp.setdefault(agg.fingerprint, []).append(agg)

    stats: list[FixedStat] = []
    groups = [("__global__", aggregates)] + sorted(by_fp.items())
    for key, aggs in groups:
        total = sum(len(a.member_lines) for a in aggs)
        suppressed = sum(a.suppressed_count for a in aggs)
        windows = len(aggs)
        fixed = floor_fixed(suppressed, total, scale)
        breakdown = {
            QUANT: 1,
            # Each window boundary may flip at most one member attribution.
            WINDOW: _ulp(windows / total, scale) if total else 0,
            LAYER: 0,
        }
        stats.append(FixedStat(
            metric="suppression_ratio", key=key, numerator=suppressed,
            denominator=total, scale_k=scale_k, fixed_value=fixed,
            truncation="floor", ulp_breakdown=breakdown,
            dominant_segment=_dominant(breakdown)))
    return stats


def root_cause_hit_rate(aggregates: list[AggregateAlert],
                        correlations: list[Correlation], scale_k: int = 4
                        ) -> FixedStat:
    scale = 10 ** scale_k
    hits = sum(1 for corr in correlations
               if any(c.direction != "self" for c in corr.candidates))
    n = len(aggregates)
    # Layer-sensitive: an aggregate with a candidate exactly at its rule's
    # max effective layer could flip under an alternative layer cutoff.
    sensitive = 0
    for corr in correlations:
        layers = [c.layer for c in corr.candidates if c.direction != "self"]
        if layers and max(layers) >= 1:
            sensitive += 1
    breakdown = {
        QUANT: 1,
        WINDOW: 0,
        LAYER: _ulp(sensitive / n, scale) if n else 0,
    }
    return FixedStat(
        metric="root_cause_hit_rate", key="__global__", numerator=hits,
        denominator=n, scale_k=scale_k,
        fixed_value=floor_fixed(hits, n, scale), truncation="floor",
        ulp_breakdown=breakdown, dominant_segment=_dominant(breakdown))


def average_aggregate_count(aggregates: list[AggregateAlert],
                            service_count: int, scale_k: int = 4
                            ) -> FixedStat:
    scale = 10 ** scale_k
    n_agg = len(aggregates)
    windows = n_agg
    breakdown = {
        QUANT: 1,
        # A boundary flip can merge/split at most one window per existing
        # window, perturbing the aggregate count by <= windows.
        WINDOW: _ulp(windows / service_count, scale) if service_count else 0,
        LAYER: 0,
    }
    return FixedStat(
        metric="avg_aggregates_per_service", key="__global__",
        numerator=n_agg, denominator=service_count, scale_k=scale_k,
        fixed_value=floor_fixed(n_agg, service_count, scale),
        truncation="floor", ulp_breakdown=breakdown,
        dominant_segment=_dominant(breakdown))


def rank_fingerprints(stats: list[FixedStat]) -> list[FixedStat]:
    """Stable ranking with synthetic-error-bound indistinguishability flag.

    Two fingerprints are indistinguishable when their fixed-point difference is
    within the larger of their synthetic ULP bounds. Tie-break: higher fixed
    value first, then fingerprint lexicographically ascending; the affected
    entries are flagged ``error_indistinguishable``.
    """
    fp_stats = [s for s in stats if s.metric == "suppression_ratio"
                and s.key != "__global__"]
    ordered = sorted(fp_stats, key=lambda s: (-s.fixed_value, s.key))
    flagged: set[str] = set()
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if left.fixed_value != right.fixed_value and \
                    (left.fixed_value - right.fixed_value
                     > max(left.total_ulp, right.total_ulp)):
                continue
            if abs(left.fixed_value - right.fixed_value) <= \
                    max(left.total_ulp, right.total_ulp):
                flagged.add(left.key)
                flagged.add(right.key)
    from dataclasses import replace
    return [replace(stat, error_indistinguishable=stat.key in flagged)
            for stat in ordered]

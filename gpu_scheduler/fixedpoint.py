"""Integer fixed-point statistics with explicit truncation and error bounds.

All statistics are stored as integers scaled by a session supplied power-of-ten
``scale`` (for example ``scale=10000`` means a ratio of ``0.1234`` is stored as
``1234``).  Ratios use deterministic *toward-zero* truncation and average wait
uses deterministic *floor* truncation.  Every returned value is paired with an
explicit error bound so callers know the maximum absolute quantization error.

The module only uses integer arithmetic; given the same inputs it always
returns byte-for-byte identical integers, independent of float hashing or dict
iteration order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Truncation directions.  They are stored verbatim in the session snapshot so
# downstream consumers never have to guess how a value was derived.
TRUNC_TOWARD_ZERO = "TOWARD_ZERO"
TRUNC_FLOOR = "FLOOR"


def _scale_factor(scale: int) -> int:
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    return scale


def quantize_ratio(numerator: int, denominator: int, scale: int) -> int:
    """Return ``numerator / denominator`` scaled by ``scale`` (toward zero).

    A zero denominator is defined as a zero ratio rather than an error, because
    empty resource pools and empty windows are legitimate in the samples.
    """
    _scale_factor(scale)
    if denominator == 0:
        return 0
    # Both operands are non-negative counters so toward-zero and floor coincide,
    # but the explicit sign handling keeps the contract obvious if reused.
    quotient, remainder = divmod(abs(numerator) * scale, abs(denominator))
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * quotient


def quantize_average(total: int, count: int, scale: int) -> int:
    """Return ``total / count`` scaled by ``scale`` using floor truncation."""
    _scale_factor(scale)
    if count <= 0:
        return 0
    # Floor division for non-negative totals; totals here are wait milliseconds.
    return total * scale // count


@dataclass(frozen=True)
class RatioStat:
    """A scaled ratio plus its explicit absolute error bound."""

    value: int
    scale: int
    truncation: str
    abs_error_bound: int  # in the same scaled unit as ``value``

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "scale": self.scale,
            "truncation": self.truncation,
            "abs_error_bound": self.abs_error_bound,
        }


def ratio_stat(numerator: int, denominator: int, scale: int) -> RatioStat:
    value = quantize_ratio(numerator, denominator, scale)
    # |true*scale - truncated| < 1 scaled unit, therefore the tight integer
    # bound expressed in scaled units is 1 (strictly less than one unit).
    bound = 1 if denominator > 0 else 0
    return RatioStat(value, scale, TRUNC_TOWARD_ZERO, bound)


@dataclass(frozen=True)
class AverageStat:
    value: int
    scale: int
    truncation: str
    abs_error_bound: int

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "scale": self.scale,
            "truncation": self.truncation,
            "abs_error_bound": self.abs_error_bound,
        }


def average_stat(total: int, count: int, scale: int) -> AverageStat:
    value = quantize_average(total, count, scale)
    bound = 1 if count > 0 else 0
    return AverageStat(value, scale, TRUNC_FLOOR, bound)


@dataclass(frozen=True)
class Comparison:
    """Result of comparing two fixed-point values under their error bounds."""

    left: int
    right: int
    difference: int
    combined_error_bound: int
    within_error: bool
    # -1 means left < right, 0 means tie, 1 means left > right.
    order: int
    tie_break_applied: bool
    tie_break_key: str

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "difference": self.difference,
            "combined_error_bound": self.combined_error_bound,
            "within_error": self.within_error,
            "order": self.order,
            "tie_break_applied": self.tie_break_applied,
            "tie_break_key": self.tie_break_key,
        }


def compare_ratios(
    left: RatioStat,
    right: RatioStat,
    left_key: str,
    right_key: str,
) -> Comparison:
    """Compare two ratios, applying a stable tie break inside the error band.

    Each value carries an absolute error bound of one scaled unit.  The
    combined band of their difference is therefore two scaled units.  When the
    integer difference falls inside ``[-2, 2]`` the result is flagged as
    error-affected and ordering falls back to the lexicographically smaller
    identifier so repeated runs always converge to the same ranking.
    """
    if left.scale != right.scale:
        raise ValueError("comparison requires a common scale")
    difference = left.value - right.value
    combined = left.abs_error_bound + right.abs_error_bound
    within_error = abs(difference) <= combined
    if not within_error:
        order = -1 if difference < 0 else 1
        tie_break = False
        tie_key = ""
    else:
        # Stable, error-affected decision: larger value still wins on a strict
        # integer difference, exact equality breaks to the smaller identifier.
        if difference != 0:
            order = -1 if difference < 0 else 1
        else:
            order = -1 if left_key <= right_key else 1
        tie_break = True
        tie_key = min(left_key, right_key)
    return Comparison(
        left=left.value,
        right=right.value,
        difference=difference,
        combined_error_bound=combined,
        within_error=within_error,
        order=order,
        tie_break_applied=tie_break,
        tie_break_key=tie_key,
    )


def aggregate_ratio(
    parts: Sequence[Tuple[int, int]],
    scale: int,
) -> RatioStat:
    """Aggregate ``(numerator, denominator)`` parts into one fixed ratio.

    Aggregation is performed on the exact integer totals *before* scaling, so
    the only quantization error is the single final truncation and the
    aggregated absolute error bound stays at one scaled unit.  This is strictly
    tighter than averaging already-truncated per-pool values, which would
    accumulate an error of up to ``len(parts)`` scaled units.
    """
    total_num = 0
    total_den = 0
    for numerator, denominator in parts:
        if numerator < 0 or denominator < 0:
            raise ValueError("aggregate parts must be non-negative")
        total_num += numerator
        total_den += denominator
    return ratio_stat(total_num, total_den, scale)


def rank(
    items: List[Tuple[str, RatioStat]],
) -> List[dict]:
    """Return a deterministic ranking of ``(key, ratio)`` items.

    Higher ratios rank first; comparisons inside the combined error band are
    marked ``error_affected`` and exact-equal values break on the key.  The
    output order is stable across runs because every sort key is an integer or
    a string, never a hash seed.
    """
    ordered: List[Tuple[str, RatioStat]] = list(items)

    def sort_key(entry: Tuple[str, RatioStat]) -> Tuple[int, str]:
        key, stat = entry
        return (-stat.value, key)

    ordered.sort(key=sort_key)
    result = []
    for position, (key, stat) in enumerate(ordered):
        flags = {"rank": position, "key": key, "stat": stat.to_dict(),
                 "error_affected_against": []}
        for other_key, other_stat in ordered:
            if other_key == key:
                continue
            comparison = compare_ratios(stat, other_stat, key, other_key)
            if comparison.within_error:
                flags["error_affected_against"].append(other_key)
        result.append(flags)
    return result

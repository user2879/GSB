"""Fixed-point share quantization and explicit quantization-error bounds.

A frame's share is the rational ``weight / total_weight``. With an integer
scale factor ``S`` the fixed-point representation is::

    q = floor(weight * S / total_weight)        # integer, 0 <= q <= S

Each truncation introduces a per-frame absolute error strictly below one
fixed-point unit::

    real_share - q/S in [0, 1/S)                 # error bound = 1/S

All ranking decisions are additionally computed with exact integer/rational
arithmetic, so the reported top-K is the *true* top-K; the fixed-point values
only drive display and the explicit "error-affected" annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Tuple

from .errors import RangeError
from .folding import FoldResult
from .ranking import FrameRank


@dataclass(frozen=True)
class QuantParams:
    scale: int  # fixed-point scale factor, e.g. 10_000 == basis points*100

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            raise RangeError(
                "scale must be an integer", field="scale")
        if self.scale <= 0:
            raise RangeError(
                f"scale must be a positive integer, got {self.scale}",
                field="scale",
            )


@dataclass(frozen=True)
class FrameShare:
    frame: str
    weight_kind: str             # "self" or "cumulative"
    weight: int
    quantized: int               # floor(weight * scale / total_weight)
    real_numerator: int          # exact rational share numerator (weight)
    real_denominator: int        # exact rational share denominator (total)
    error_bound_num: int         # error bound as a fraction: 1 / scale
    error_bound_den: int
    error_affected: bool         # quantized value ties/indistinguishable vs peer


@dataclass(frozen=True)
class ShareReport:
    scale: int
    total_weight: int
    self_shares: Tuple[FrameShare, ...]
    cumulative_shares: Tuple[FrameShare, ...]
    aggregate_error_bound_num: int   # m/S for the top-K set size m
    aggregate_error_bound_den: int
    top_k_contains_true_top_k: bool
    top_k_basis: str
    error_affected_pairs: Tuple[Tuple[str, str, str], ...]


def quantize(weight: int, total_weight: int, scale: int) -> int:
    if total_weight <= 0:
        raise RangeError(
            "total_weight must be positive to compute shares",
            field="total_weight")
    return (weight * scale) // total_weight


def _build_shares(metrics_weight: Sequence[Tuple[str, int]],
                  weight_kind: str, total_weight: int,
                  scale: int) -> Tuple[List[FrameShare], set]:
    shares: List[FrameShare] = []
    indistinguishable_frames: set[str] = set()
    weight_by_frame = dict(metrics_weight)
    quantized_by_frame = {
        frame: quantize(weight, total_weight, scale)
        for frame, weight in metrics_weight
    }
    # Two frames are quantization-indistinguishable when their fixed-point
    # values differ by at most one unit (the synthetic error envelope 1/S).
    ordered_frames = [frame for frame, _ in metrics_weight]
    for index, frame in enumerate(ordered_frames):
        q_here = quantized_by_frame[frame]
        affected = False
        for other in ordered_frames:
            if other == frame:
                continue
            if abs(q_here - quantized_by_frame[other]) <= 1:
                affected = True
                break
        if affected:
            indistinguishable_frames.add(frame)
        shares.append(
            FrameShare(
                frame=frame,
                weight_kind=weight_kind,
                weight=weight_by_frame[frame],
                quantized=q_here,
                real_numerator=weight_by_frame[frame],
                real_denominator=total_weight,
                error_bound_num=1,
                error_bound_den=scale,
                error_affected=affected,
            )
        )
    return shares, indistinguishable_frames


def build_share_report(result: FoldResult,
                       ranked: Sequence[FrameRank],
                       params: QuantParams) -> ShareReport:
    """Build fixed-point shares plus error annotations for ranked frames."""

    scale = params.scale
    total_weight = result.total_weight
    top_k = len(ranked)
    ranked_frames = [entry.frame for entry in ranked]

    metric_map = result.metric_map()
    self_pairs = [
        (frame, metric_map[frame].self_weight) for frame in ranked_frames
    ]
    cumulative_pairs = [
        (frame, metric_map[frame].cumulative_weight)
        for frame in ranked_frames
    ]

    self_shares, self_affected = _build_shares(
        self_pairs, "self", total_weight, scale)
    cumulative_shares, cumulative_affected = _build_shares(
        cumulative_pairs, "cumulative", total_weight, scale)

    # True top-K decided with exact rational arithmetic on self weight, using
    # cumulative then name as deterministic tie-breakers (same rule as rank).
    all_self = [
        (metric.frame, metric.self_weight)
        for metric in result.frame_metrics
    ]
    all_cumulative = {
        metric.frame: metric.cumulative_weight
        for metric in result.frame_metrics
    }
    exact_order = sorted(
        all_self,
        key=lambda item: (
            -Fraction(item[1], total_weight),
            -Fraction(all_cumulative[item[0]], total_weight),
            item[0],
        ),
    )
    true_set = {frame for frame, _ in exact_order[:top_k]}
    reported_set = set(ranked_frames)
    contains_true = true_set <= reported_set

    affected_pairs: List[Tuple[str, str, str]] = []
    for index in range(len(ranked_frames) - 1):
        left, right = ranked_frames[index], ranked_frames[index + 1]
        if left in self_affected and right in self_affected:
            affected_pairs.append((left, right, "self"))
        if left in cumulative_affected and right in cumulative_affected:
            affected_pairs.append((left, right, "cumulative"))

    # Aggregate (sum-of-top-K) error bound: m independent truncations each
    # strictly below 1/S, so the bound is m/S.
    return ShareReport(
        scale=scale,
        total_weight=total_weight,
        self_shares=tuple(self_shares),
        cumulative_shares=tuple(cumulative_shares),
        aggregate_error_bound_num=top_k,
        aggregate_error_bound_den=scale,
        top_k_contains_true_top_k=contains_true,
        top_k_basis=(
            "exact rational compare of weight/total_weight with "
            "(-self,-cumulative,name) total order; fixed-point values are "
            "display-only and never decide membership"
        ),
        error_affected_pairs=tuple(affected_pairs),
    )

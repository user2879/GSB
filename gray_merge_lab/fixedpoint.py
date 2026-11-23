from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import is_int
from .errors import LabError


TRUNCATION = "toward_zero"


def validate_scale_exp(scale_exp: Any) -> int:
    if not is_int(scale_exp) or scale_exp < 0 or scale_exp > 9:
        raise LabError("E_SPEC", "scale_exp must be an integer in 0..9", field="scale_exp")
    return int(scale_exp)


def trunc_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


@dataclass(frozen=True)
class BatchFixedStats:
    batch_id: str
    raw_duration_ms: int
    raw_samples: int
    scale_exp: int

    def __post_init__(self) -> None:
        if self.raw_duration_ms < 0 or self.raw_samples < 0:
            raise LabError("E_FORMAT", "duration and sample count must be non-negative")
        validate_scale_exp(self.scale_exp)

    @property
    def scale(self) -> int:
        return 10**self.scale_exp

    @property
    def duration_q(self) -> int:
        return self.raw_duration_ms * self.scale

    @property
    def samples_q(self) -> int:
        return self.raw_samples * self.scale

    @property
    def single_value_error_original(self) -> float:
        return 1.0 / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "raw_duration_ms": self.raw_duration_ms,
            "raw_samples": self.raw_samples,
            "scale_exp": self.scale_exp,
            "scale": self.scale,
            "truncation": TRUNCATION,
            "duration_q": self.duration_q,
            "samples_q": self.samples_q,
            "single_value_error_original": self.single_value_error_original,
            "single_value_error_q": 1,
        }


@dataclass(frozen=True)
class AggregateFixedStats:
    batch_ids: tuple[str, ...]
    raw_duration_sum_ms: int
    raw_sample_sum: int
    batch_count: int
    scale_exp: int

    @property
    def scale(self) -> int:
        return 10**self.scale_exp

    @property
    def avg_duration_q(self) -> int:
        return trunc_div(self.raw_duration_sum_ms * self.scale, self.batch_count)

    @property
    def avg_samples_q(self) -> int:
        return trunc_div(self.raw_sample_sum * self.scale, self.batch_count)

    @property
    def single_value_error_original(self) -> float:
        return 1.0 / self.scale

    @property
    def batch_aggregate_error_original(self) -> float:
        # Worst case if independently quantized batch means are re-aggregated.
        return self.batch_count / self.scale

    @property
    def canonical_aggregate_error_original(self) -> float:
        # This implementation sums raw integers first and truncates once.
        return 1.0 / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_ids": list(self.batch_ids),
            "batch_count": self.batch_count,
            "scale_exp": self.scale_exp,
            "scale": self.scale,
            "truncation": TRUNCATION,
            "raw_duration_sum_ms": self.raw_duration_sum_ms,
            "raw_sample_sum": self.raw_sample_sum,
            "avg_duration_q": self.avg_duration_q,
            "avg_samples_q": self.avg_samples_q,
            "avg_duration_original_approx": self.avg_duration_q / self.scale,
            "avg_samples_original_approx": self.avg_samples_q / self.scale,
            "single_value_error_bound_original": self.single_value_error_original,
            "batch_aggregate_error_bound_original": self.batch_aggregate_error_original,
            "canonical_aggregate_error_bound_original": self.canonical_aggregate_error_original,
            "single_value_error_bound_q": 1,
            "batch_aggregate_error_bound_q": self.batch_count,
            "canonical_aggregate_error_bound_q": 1,
        }


def aggregate_batch_stats(stats: Iterable[BatchFixedStats]) -> AggregateFixedStats:
    items = tuple(sorted(stats, key=lambda item: item.batch_id))
    if not items:
        raise LabError("E_FORMAT", "cannot aggregate zero batches")
    scale = items[0].scale_exp
    if any(item.scale_exp != scale for item in items):
        raise LabError("E_FORMAT", "all batches must use the same scale_exp", field="scale_exp")
    return AggregateFixedStats(
        batch_ids=tuple(item.batch_id for item in items),
        raw_duration_sum_ms=sum(item.raw_duration_ms for item in items),
        raw_sample_sum=sum(item.raw_samples for item in items),
        batch_count=len(items),
        scale_exp=scale,
    )


def q_to_original_interval(q_value: int, scale_exp: int) -> tuple[float, float]:
    validate_scale_exp(scale_exp)
    scale = 10**scale_exp
    return (q_value / scale, (q_value + 1) / scale)


def raw_duration_to_q(raw_duration_ms: int, scale_exp: int) -> int:
    if raw_duration_ms < 0:
        raise ValueError("duration must be non-negative")
    validate_scale_exp(scale_exp)
    return raw_duration_ms * (10**scale_exp)


@dataclass(frozen=True)
class CandidateDecision:
    winner: str
    indistinguishable: bool
    error_affected: bool
    difference_q: int
    combined_error_bound_q: int
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "indistinguishable": self.indistinguishable,
            "error_affected": self.error_affected,
            "difference_q": self.difference_q,
            "combined_error_bound_q": self.combined_error_bound_q,
            "rule": self.rule,
        }


def compare_candidates(
    left_batch_id: str,
    left_avg_duration_q: int,
    right_batch_id: str,
    right_avg_duration_q: int,
) -> CandidateDecision:
    """Compare two fixed-point average durations.

    Each canonical average truncates once, so each has error strictly below one
    scaled integer. The combined bound is therefore two scaled integers. Values
    within the conservative inclusive bound are indistinguishable and are
    resolved lexicographically by batch id.
    """
    difference = abs(left_avg_duration_q - right_avg_duration_q)
    combined = 2
    if difference <= combined:
        return CandidateDecision(
            winner=min(left_batch_id, right_batch_id),
            indistinguishable=True,
            error_affected=True,
            difference_q=difference,
            combined_error_bound_q=combined,
            rule="within_combined_error_bound:lexicographic_batch_id",
        )
    winner = left_batch_id if left_avg_duration_q < right_avg_duration_q else right_batch_id
    return CandidateDecision(
        winner=winner,
        indistinguishable=False,
        error_affected=False,
        difference_q=difference,
        combined_error_bound_q=combined,
        rule="outside_combined_error_bound:lower_avg_duration",
    )


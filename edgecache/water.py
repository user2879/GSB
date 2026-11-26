"""Water-level error attribution and deterministic candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .spec import RATE_SCALE, GlobalSpec


MICRO = 1_000_000
LEVEL_WEIGHT_QUANTIZATION = "block_weight_quantization"
LEVEL_SAMPLING_COVERAGE = "tiered_sampling_coverage"
LEVEL_EVICTION_ACCUMULATION = "eviction_count_accumulation"


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


@dataclass(frozen=True)
class ErrorAttribution:
    total_blocks: int
    uncovered_blocks: int
    unknown_batches: int
    block_weight: int
    quantization_ppm: int
    contributions_micro: dict[str, int]
    bound_micro: int
    dominant_level: str
    dominant_threshold_ppm: int
    affected_by_error: bool

    def to_dict(self) -> dict[str, Any]:
        ratios = {
            level: (contribution * RATE_SCALE // self.bound_micro if self.bound_micro else 0)
            for level, contribution in self.contributions_micro.items()
        }
        return {
            "total_blocks": self.total_blocks,
            "uncovered_blocks": self.uncovered_blocks,
            "unknown_batches": self.unknown_batches,
            "block_weight": self.block_weight,
            "quantization_ppm": self.quantization_ppm,
            "unit": "micro_quota",
            "contributions_micro": dict(sorted(self.contributions_micro.items())),
            "ratios_ppm": dict(sorted(ratios.items())),
            "bound_micro": self.bound_micro,
            "bound_quota": str(Fraction(self.bound_micro, MICRO)),
            "dominant_level": self.dominant_level,
            "dominant_threshold_ppm": self.dominant_threshold_ppm,
            "dominant_decision": "dominant" if self.dominant_level != "none" else "none",
            "affected_by_error": self.affected_by_error,
        }


def attribute_water_error(
    total_blocks: int,
    uncovered_blocks: int,
    unknown_batches: int,
    global_spec: GlobalSpec,
) -> ErrorAttribution:
    if total_blocks < 0 or uncovered_blocks < 0 or uncovered_blocks > total_blocks:
        raise ValueError("invalid block coverage counts")
    if unknown_batches < 0:
        raise ValueError("unknown_batches must be non-negative")
    quant_micro = _ceil_fraction(
        Fraction(total_blocks * global_spec.block_weight * global_spec.block_weight_quantum_ppm, RATE_SCALE)
    )
    coverage_micro = uncovered_blocks * global_spec.block_weight * MICRO
    accumulation_micro = (
        unknown_batches
        * global_spec.max_unknown_batch_evictions
        * global_spec.block_weight
        * MICRO
    )
    contributions = {
        LEVEL_WEIGHT_QUANTIZATION: quant_micro,
        LEVEL_SAMPLING_COVERAGE: coverage_micro,
        LEVEL_EVICTION_ACCUMULATION: accumulation_micro,
    }
    bound_micro = sum(contributions.values())
    dominant = "none"
    if bound_micro:
        for level, contribution in contributions.items():
            if contribution * RATE_SCALE >= global_spec.dominant_threshold_ppm * bound_micro:
                if dominant == "none" or contribution > contributions[dominant]:
                    dominant = level
    return ErrorAttribution(
        total_blocks=total_blocks,
        uncovered_blocks=uncovered_blocks,
        unknown_batches=unknown_batches,
        block_weight=global_spec.block_weight,
        quantization_ppm=global_spec.block_weight_quantum_ppm,
        contributions_micro=contributions,
        bound_micro=bound_micro,
        dominant_level=dominant,
        dominant_threshold_ppm=global_spec.dominant_threshold_ppm,
        affected_by_error=False,
    )


@dataclass(frozen=True)
class WaterCandidate:
    batch_id: str
    water_level: int
    evict_blocks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "water_level": self.water_level,
            "evict_blocks": self.evict_blocks,
        }


@dataclass(frozen=True)
class CandidateDecision:
    chosen: WaterCandidate
    rejected: WaterCandidate
    absolute_water_diff: int
    error_bound_micro: int
    affected_by_error: bool
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen.to_dict(),
            "rejected": self.rejected.to_dict(),
            "absolute_water_diff": self.absolute_water_diff,
            "error_bound_micro": self.error_bound_micro,
            "water_diff_micro": self.absolute_water_diff * MICRO,
            "affected_by_error": self.affected_by_error,
            "rule": self.rule,
        }


def select_water_candidate(
    first: WaterCandidate,
    second: WaterCandidate,
    error_bound_micro: int,
) -> CandidateDecision:
    if error_bound_micro < 0:
        raise ValueError("error_bound_micro must be non-negative")
    absolute_diff = abs(first.water_level - second.water_level)
    within_bound = absolute_diff * MICRO <= error_bound_micro
    if within_bound:
        if first.water_level <= second.water_level:
            chosen, rejected = first, second
        elif second.water_level < first.water_level:
            chosen, rejected = second, first
        else:
            chosen, rejected = sorted((first, second), key=lambda item: item.batch_id)
        rule = "within_bound_lower_water_then_batch_id"
    else:
        chosen, rejected = sorted((first, second), key=lambda item: (item.water_level, item.batch_id))
        rule = "outside_bound_lower_water_then_batch_id"
    return CandidateDecision(
        chosen=chosen,
        rejected=rejected,
        absolute_water_diff=absolute_diff,
        error_bound_micro=error_bound_micro,
        affected_by_error=within_bound,
        rule=rule,
    )


def aggregate_unknown_evictions(unknown_records: Iterable[dict[str, Any]], global_spec: GlobalSpec) -> int:
    total = 0
    for record in unknown_records:
        evictions = record.get("evict_blocks", global_spec.max_unknown_batch_evictions)
        if isinstance(evictions, int) and evictions >= 0:
            total += evictions
    return total


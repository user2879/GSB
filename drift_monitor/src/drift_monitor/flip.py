"""Quantization tier switching and judgement-flip diagnostics.

For every (column, window) we evaluate the raw metric score on every
configured tier plus an implicit reference tier two decimals wider than the
finest one.  Adjacent tier pairs are compared stage by stage
(quantize -> threshold -> judgement); the first stage whose fixed-point
difference exceeds the tolerance is recorded as ``first_breach_stage``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fixedpoint import (
    QuantTier,
    is_significant,
    quantize,
    quantize_threshold,
    reference_tier,
    retained,
)

STAGES = ("quantize", "threshold", "judgement")


@dataclass(frozen=True)
class TierEvaluation:
    tier: str
    decimals: int
    score_q: int
    score_retained: float
    threshold_q: int
    significant: bool


@dataclass(frozen=True)
class FlipRecord:
    column: str
    window_index: int
    lower_tier: str
    higher_tier: str
    lower_score_q: int
    higher_score_q: int
    lower_retained: float
    higher_retained: float
    raw_score: float
    absolute_q_diff: int
    tolerance_q: int
    first_breach_stage: str | None
    flipped: bool
    reference_score_q: int
    reference_retained: float
    reference_tier: str


def evaluate_tiers(raw_score: float, threshold: float,
                   tiers: tuple[QuantTier, ...]) -> tuple[TierEvaluation, ...]:
    results = []
    for tier in tiers:
        score_q = quantize(raw_score, tier)
        threshold_q = quantize_threshold(threshold, tier)
        results.append(
            TierEvaluation(
                tier=tier.name,
                decimals=tier.decimals,
                score_q=score_q,
                score_retained=retained(score_q, tier),
                threshold_q=threshold_q,
                significant=is_significant(score_q, threshold_q),
            )
        )
    return tuple(results)


def diagnose_flips(column: str, window_index: int, raw_score: float,
                   threshold: float, tiers: tuple[QuantTier, ...],
                   tolerance_fraction: float = 0.001) -> tuple[
    tuple[TierEvaluation, ...], tuple[FlipRecord, ...], TierEvaluation
]:
    evaluations = evaluate_tiers(raw_score, threshold, tiers)
    ref_tier = reference_tier(tiers)
    ref_score_q = quantize(raw_score, ref_tier)
    ref_eval = TierEvaluation(
        tier=ref_tier.name,
        decimals=ref_tier.decimals,
        score_q=ref_score_q,
        score_retained=retained(ref_score_q, ref_tier),
        threshold_q=quantize_threshold(threshold, ref_tier),
        significant=is_significant(ref_score_q, quantize_threshold(threshold, ref_tier)),
    )
    flips = []
    for idx in range(len(evaluations) - 1):
        lower, higher = evaluations[idx], evaluations[idx + 1]
        lower_tier, higher_tier = tiers[idx], tiers[idx + 1]
        q_diff = abs(higher.score_q - lower.score_q *
                     10 ** (higher_tier.decimals - lower_tier.decimals))
        tolerance_q = max(1, int(round(tolerance_fraction *
                                       10 ** higher_tier.decimals)))
        first_breach = None
        if higher.significant != lower.significant:
            first_breach = "judgement"
        elif q_diff > tolerance_q:
            first_breach = "quantize"
        elif abs(higher.threshold_q - lower.threshold_q *
                 10 ** (higher_tier.decimals - lower_tier.decimals)) > tolerance_q:
            first_breach = "threshold"
        flips.append(
            FlipRecord(
                column=column,
                window_index=window_index,
                lower_tier=lower.tier,
                higher_tier=higher.tier,
                lower_score_q=lower.score_q,
                higher_score_q=higher.score_q,
                lower_retained=lower.score_retained,
                higher_retained=higher.score_retained,
                raw_score=raw_score,
                absolute_q_diff=q_diff,
                tolerance_q=tolerance_q,
                first_breach_stage=first_breach,
                flipped=higher.significant != lower.significant,
                reference_score_q=ref_score_q,
                reference_retained=retained(ref_score_q, ref_tier),
                reference_tier=ref_tier.name,
            )
        )
    return evaluations, tuple(flips), ref_eval

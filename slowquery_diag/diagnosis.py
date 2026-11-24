"""Stable final adjudication when candidates are indistinguishable.

Two candidate rewrites are compared on (a) their fixed-point evidence
latency and (b) their fixed-point expected gain.  When an absolute
difference falls inside the synthesized decision error bound the pair is
marked error-affected and broken by an explicit total order::

    1. candidate identifier lexicographic ascending
    2. the minimum source-type bias rank among target sources

Cause attribution distinguishes three categories and reports ratios:
    * error_indistinguishable: differences inside the decision error bound
    * caliber_conflict: participating evidence uses different latency calibers
    * high_rejected_evidence: rejected / total evidence ratio >= threshold
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import CandidateRule, SOURCE_TYPE_BIAS
from .quantization import ErrorBound, Quantizer, indistinguishable

REJECTED_RATIO_THRESHOLD = 0.5
_QUERY_ID_PATTERN = re.compile(r"\d+")


def query_id_of(candidate: CandidateRule) -> int:
    match = _QUERY_ID_PATTERN.search(candidate.original_query)
    return int(match.group(0)) if match else 0


def candidate_order_key(candidate: CandidateRule,
                        source_type_by_id: dict[str, str] | None = None
                        ) -> tuple[str, int]:
    source_type_by_id = source_type_by_id or {}
    min_bias = (
        min(
            SOURCE_TYPE_BIAS.get(source_type_by_id.get(s, ""),
                                 len(SOURCE_TYPE_BIAS))
            for s in candidate.target_sources
        )
        if candidate.target_sources
        else len(SOURCE_TYPE_BIAS)
    )
    return (candidate.candidate_id, min_bias)


@dataclass
class CandidateEvaluation:
    candidate_id: str
    query_id: int
    target_sources: tuple[str, ...]
    baseline_latency_ticks: int
    expected_gain_ticks: int
    replay_latency_ticks: int | None = None
    replay_rows_ticks: int | None = None
    error_affected: bool = False
    decision_note: str = ""


@dataclass
class CauseAttribution:
    query_id: int
    rejected_ratio: float
    caliber_conflict: bool
    error_indistinguishable: bool
    causes: tuple[str, ...] = ()


@dataclass
class DiagnosisReport:
    scale_exponent: int
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    attributions: list[CauseAttribution] = field(default_factory=list)
    session_ratios: dict[str, float] = field(default_factory=dict)


def _rejected_ratio(query_verdict) -> float:
    views = list(query_verdict.latency.views) + list(query_verdict.rows.views)
    if not views:
        return 0.0
    rejected = sum(1 for view in views if not view.adopted)
    return rejected / len(views)


def build_report(
    candidates: list[CandidateRule],
    query_verdicts: dict,
    quantizer: Quantizer,
    error_bounds: dict[tuple[str, int], ErrorBound],
    source_type_by_id: dict[str, str],
) -> DiagnosisReport:
    evaluations: list[CandidateEvaluation] = []
    attributions: list[CauseAttribution] = []

    for candidate in sorted(
        candidates,
        key=lambda item: candidate_order_key(item, source_type_by_id),
    ):
        query_id = query_id_of(candidate)
        verdict = query_verdicts.get(query_id)
        if verdict is None:
            continue
        bound = error_bounds.get(("latency_us", query_id))

        baseline_ticks = quantizer.quantize(verdict.latency.adopted_value)
        gain_ticks = quantizer.quantize(candidate.expected_gain_bps)
        evaluations.append(
            CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                query_id=query_id,
                target_sources=candidate.target_sources,
                baseline_latency_ticks=baseline_ticks,
                expected_gain_ticks=gain_ticks,
            )
        )

        caliber_conflict = bound is not None and bound.caliber_raw > 0
        ratio = _rejected_ratio(verdict)
        causes: list[str] = []
        if ratio >= REJECTED_RATIO_THRESHOLD:
            causes.append("high_rejected_evidence")
        if caliber_conflict:
            causes.append("caliber_conflict")
        attributions.append(
            CauseAttribution(
                query_id=query_id,
                rejected_ratio=round(ratio, 6),
                caliber_conflict=caliber_conflict,
                error_indistinguishable=False,
                causes=tuple(causes),
            )
        )

    _stable_pairwise_tiebreak(evaluations, error_bounds)
    _finalize_attributions(evaluations, attributions)
    total = max(len(attributions), 1)
    session_ratios = {
        "error_indistinguishable": sum(
            1 for attr in attributions if attr.error_indistinguishable
        ) / total,
        "caliber_conflict": sum(
            1 for attr in attributions if attr.caliber_conflict
        ) / total,
        "high_rejected_evidence": sum(
            1 for attr in attributions
            if attr.rejected_ratio >= REJECTED_RATIO_THRESHOLD
        ) / total,
    }
    return DiagnosisReport(
        scale_exponent=quantizer.scale_exponent,
        evaluations=evaluations,
        attributions=attributions,
        session_ratios=session_ratios,
    )


def _stable_pairwise_tiebreak(
    evaluations: list[CandidateEvaluation],
    error_bounds: dict[tuple[str, int], ErrorBound],
) -> None:
    groups: dict[int, list[CandidateEvaluation]] = {}
    for evaluation in evaluations:
        groups.setdefault(evaluation.query_id, []).append(evaluation)

    for query_id, group in groups.items():
        bound = error_bounds.get(("latency_us", query_id))
        bound_ticks = bound.total_ticks if bound else 0
        group.sort(key=lambda item: item.candidate_id)
        if len(group) < 2:
            group[0].decision_note = "single candidate; adopted directly"
            continue
        best = group[0]
        for other in group[1:]:
            latency_tie = indistinguishable(
                best.baseline_latency_ticks,
                other.baseline_latency_ticks,
                bound_ticks,
            )
            gain_tie = indistinguishable(
                best.expected_gain_ticks,
                other.expected_gain_ticks,
                max(bound_ticks, 1),
            )
            if latency_tie or gain_tie:
                other.error_affected = True
                best.error_affected = True
                other.decision_note = (
                    f"error-indistinguishable from {best.candidate_id}; "
                    "stable tie-break keeps lower candidate_id/source-bias"
                )
            else:
                other.decision_note = "distinguishable; normal ranking applies"
        best.decision_note = (
            "stable winner (candidate_id then source-bias order)"
            if best.error_affected
            else "leading candidate"
        )


def _finalize_attributions(
    evaluations: list[CandidateEvaluation],
    attributions: list[CauseAttribution],
) -> None:
    by_query = {attr.query_id: attr for attr in attributions}
    for evaluation in evaluations:
        attr = by_query.get(evaluation.query_id)
        if attr is not None and evaluation.error_affected:
            attr.error_indistinguishable = True
    for attr in attributions:
        causes = set(attr.causes)
        if attr.error_indistinguishable:
            causes.add("error_indistinguishable")
        attr.causes = tuple(sorted(causes))

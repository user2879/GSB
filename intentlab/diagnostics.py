"""Per-query error propagation and decision-flip attribution."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from .models import IntentScore, MatchResult, Query, RuntimeConfig, ScoreResult, Segment, TriggerEntry
from .scoring import FloatScoreResult


def _decision(value: int, threshold: int, tie_rule: str) -> bool:
    if value == threshold:
        return tie_rule == "accept_on_tie"
    return value > threshold


def float_as_score_result(
    fixed: ScoreResult,
    reference: FloatScoreResult,
    config: RuntimeConfig,
) -> ScoreResult:
    scale = 1 << config.fraction_bits
    scores = tuple(
        IntentScore(
            intent_id=item.intent_id,
            raw_score=round(item.score * scale),
            score=round(item.score * scale),
            threshold=round(item.threshold * scale),
            decision=item.decision,
            boundary=abs(item.margin) <= (config.quantum / 2.0) / scale,
            margin=round(item.margin * scale),
        )
        for item in reference.scores
    )
    return replace(
        fixed,
        predicted_intent=reference.predicted_intent,
        scores=scores,
        boundary=any(item.boundary for item in scores),
        margin=round(reference.margin * scale),
    )


def _segment_for_query(segments: Sequence[Segment], query_id: str) -> Segment | None:
    for segment in segments:
        if query_id in segment.query_ids:
            return segment
    return None


def _slots_for_query(segment: Segment | None, query_id: str) -> tuple[tuple[str, int, int, str], ...]:
    if segment is None:
        return ()
    return tuple(
        (slot.name, slot.start, slot.end, slot.status)
        for slot in segment.aligned_slots
        if slot.query_id == query_id
    )


def _selected_signature(match: MatchResult) -> set[tuple[str, int, int]]:
    return {(span.trigger_id, span.start, span.end) for span in match.selected}


def diagnose_query(
    *,
    query: Query,
    match: MatchResult,
    fixed: ScoreResult,
    float_reference: FloatScoreResult,
    high_width: ScoreResult,
    entries: Iterable[TriggerEntry],
    config: RuntimeConfig,
    fixed_segment: Segment | None,
    reference_segment: Segment | None,
) -> dict[str, object]:
    entries = tuple(entries)
    fixed_sig = _selected_signature(match)
    reference_sig = _selected_signature(match)
    trigger_delta_count = len(fixed_sig.symmetric_difference(reference_sig))
    trigger_denominator = max(1, len(fixed_sig | reference_sig))
    trigger_delta = trigger_delta_count / trigger_denominator

    scale = 1 << config.fraction_bits
    float_by_intent = {item.intent_id: item for item in float_reference.scores}
    score_delta = 0.0
    boundary_delta = 0.0
    for item in fixed.scores:
        ref = float_by_intent.get(item.intent_id)
        if ref is None:
            continue
        score_delta = max(score_delta, abs(item.score / scale - ref.score))
        boundary_delta = max(boundary_delta, abs(item.margin / scale - ref.margin))
    boundary_denominator = max(
        1.0,
        max((abs(item.threshold) for item in fixed.scores), default=1) / scale,
    )

    fixed_slots = _slots_for_query(fixed_segment, query.query_id)
    reference_slots = _slots_for_query(reference_segment, query.query_id)
    slot_delta_count = len(set(fixed_slots).symmetric_difference(set(reference_slots)))
    slot_denominator = max(1, len(set(fixed_slots) | set(reference_slots)))
    slot_delta = slot_delta_count / slot_denominator

    fixed_membership = fixed_segment.query_ids if fixed_segment else ()
    reference_membership = reference_segment.query_ids if reference_segment else ()
    merge_changed = fixed_membership != reference_membership

    candidate_raw = next((item for item in fixed.scores if item.intent_id == fixed.predicted_intent), None)
    if candidate_raw is None and fixed.scores:
        candidate_raw = sorted(fixed.scores, key=lambda item: (-item.score, item.intent_id))[0]
    quantization_changed_decision = False
    if candidate_raw is not None:
        raw_decision = _decision(candidate_raw.raw_score, candidate_raw.threshold, config.tie_rule)
        quantization_changed_decision = raw_decision != candidate_raw.decision

    flipped = fixed.predicted_intent != float_reference.predicted_intent
    high_width_prediction = high_width.predicted_intent
    flags = {
        "quantization_rounding": bool(flipped and quantization_changed_decision),
        "threshold_comparison": bool(
            flipped and not quantization_changed_decision
        ),
        "merge_boundary_movement": bool(flipped and merge_changed),
    }
    if not flipped:
        primary_reason = "none"
    elif flags["quantization_rounding"]:
        primary_reason = "quantization_rounding"
    elif flags["threshold_comparison"]:
        primary_reason = "threshold_comparison"
    else:
        primary_reason = "merge_boundary_movement"

    stages = [
        {"stage": "trigger_match", "delta": trigger_delta, "raw_delta": trigger_delta_count},
        {"stage": "intent_score", "delta": score_delta, "raw_delta": score_delta},
        {
            "stage": "boundary_decision",
            "delta": boundary_delta / boundary_denominator,
            "raw_delta": boundary_delta,
        },
        {"stage": "slot_alignment", "delta": slot_delta, "raw_delta": slot_delta_count},
    ]
    first_exceeding = next(
        (stage["stage"] for stage in stages if float(stage["delta"]) > config.tolerance),
        None,
    )
    high_scale = 1 << high_width.fraction_bits
    return {
        "tolerance": config.tolerance,
        "stages": stages,
        "first_exceeding_stage": first_exceeding,
        "flip": {
            "fixed_intent": fixed.predicted_intent,
            "float_intent": float_reference.predicted_intent,
            "high_width_intent": high_width_prediction,
            "fixed_score": round(fixed.margin / scale, 8),
            "float_score_margin": round(float_reference.margin, 8),
            "high_width_bits": high_width.fraction_bits,
            "high_width_margin": round(high_width.margin / high_scale, 8),
            "primary_reason": primary_reason,
            "causes": flags,
        },
        "boundary": {
            "fixed": fixed.boundary,
            "float_near_threshold": any(
                abs(item.margin) <= config.quantum / 2.0 / scale for item in float_reference.scores
            ),
        },
    }

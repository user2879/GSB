from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any, Mapping

from .engine import DiagnosticEngine
from .models import (
    AppConfig,
    FixedPointConfig,
    LimitsConfig,
    Session,
)


def _real_value(units: int | None, scale: int) -> Decimal | None:
    if units is None:
        return None
    return Decimal(units) / Decimal(scale)


def _delta(actual: Decimal | None, reference: Decimal | None) -> Decimal | None:
    if actual is None or reference is None:
        return None
    return actual - reference


def _numeric_entry(
    *,
    stage: str,
    turn_id: int,
    evidence_id: str,
    actual_units: int | None,
    reference_units: int | None,
    actual_scale: int,
    reference_scale: int,
    tolerance: Decimal,
) -> dict[str, Any]:
    actual = _real_value(actual_units, actual_scale)
    reference = _real_value(reference_units, reference_scale)
    delta = _delta(actual, reference)
    exceeds = delta is not None and abs(delta) > tolerance
    return {
        "stage": stage,
        "turn_id": turn_id,
        "evidence_id": evidence_id,
        "actual_units": actual_units,
        "reference_units": reference_units,
        "actual": str(actual) if actual is not None else None,
        "reference": str(reference) if reference is not None else None,
        "delta": str(delta) if delta is not None else None,
        "exceeds_tolerance": exceeds,
    }


def _maps_by_evidence(turn: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if stage == "recall":
        payload = turn["recall"]["evidence"]
    elif stage == "dedupe":
        payload = turn["dedupe"]["evidence"]
    elif stage == "confidence":
        payload = turn["confidence"].get("citation_chain", [])
    else:
        raise ValueError(stage)
    return {item["evidence_id"]: item for item in payload}


def _confidence_value(turn: Mapping[str, Any]) -> Decimal | None:
    value = turn["confidence"].get("confidence")
    if value is None:
        return None
    return Decimal(value)


def _authority_changed(
    actual_turn: Mapping[str, Any], reference_turn: Mapping[str, Any]
) -> list[str]:
    actual = _maps_by_evidence(actual_turn, "dedupe")
    reference = _maps_by_evidence(reference_turn, "dedupe")
    changed: list[str] = []
    for evidence_id in sorted(set(actual) | set(reference)):
        if actual.get(evidence_id, {}).get("authority_channel") != reference.get(
            evidence_id, {}
        ).get("authority_channel"):
            changed.append(evidence_id)
    return changed


def _cross_topic_changed(
    actual_turn: Mapping[str, Any], reference_turn: Mapping[str, Any]
) -> list[str]:
    actual = _maps_by_evidence(actual_turn, "confidence")
    reference = _maps_by_evidence(reference_turn, "confidence")
    changed: list[str] = []
    for evidence_id in sorted(set(actual) | set(reference)):
        actual_item = actual.get(evidence_id, {})
        reference_item = reference.get(evidence_id, {})
        if actual_item.get("topic_ids") != reference_item.get("topic_ids"):
            changed.append(evidence_id)
        elif bool(actual_item.get("cross_topic")) != bool(
            reference_item.get("cross_topic")
        ):
            changed.append(evidence_id)
    return changed


def _cross_topic_contribution_delta(
    actual_turn: Mapping[str, Any],
    reference_turn: Mapping[str, Any],
    actual_scale: int,
    reference_scale: int,
) -> Decimal:
    actual = _maps_by_evidence(actual_turn, "confidence")
    reference = _maps_by_evidence(reference_turn, "confidence")
    total = Decimal(0)
    for evidence_id in sorted(set(actual) | set(reference)):
        actual_item = actual.get(evidence_id, {})
        reference_item = reference.get(evidence_id, {})
        if actual_item.get("cross_topic") or reference_item.get("cross_topic"):
            actual_value = _real_value(
                actual_item.get("contribution_units"), actual_scale
            )
            reference_value = _real_value(
                reference_item.get("contribution_units"), reference_scale
            )
            if actual_value is not None and reference_value is not None:
                total += abs(actual_value - reference_value)
    return total


def build_diagnostic_report(
    config: AppConfig, session: Session
) -> dict[str, Any]:
    actual_config = config.with_limits(
        LimitsConfig(
            max_workers=config.limits.max_workers,
            evidence_budget=max(1_000_000, len(session.evidence) * 100),
        )
    )
    reference_config = replace(
        actual_config,
        fixed_point=FixedPointConfig(
            fractional_bits=config.reference.fractional_bits,
            quantization=config.reference.quantization,
            rounding=config.reference.rounding,
        ),
    )
    actual = DiagnosticEngine(actual_config, session).compute(
        force_recompute=True
    ).result
    reference = DiagnosticEngine(reference_config, session).compute(
        force_recompute=True
    ).result
    actual_scale = int(actual["fixed_point"]["scale"])
    reference_scale = int(reference["fixed_point"]["scale"])
    tolerance = config.diagnostics.tolerance

    actual_turns = {turn["turn_id"]: turn for turn in actual["turns"]}
    reference_turns = {
        turn["turn_id"]: turn for turn in reference["turns"]
    }
    stage_entries: dict[str, list[dict[str, Any]]] = {
        "recall": [],
        "dedupe": [],
        "topic": [],
        "confidence": [],
    }
    first_exceed_by_turn: dict[str, dict[str, Any]] = {}
    flips: list[dict[str, Any]] = []
    stage_order = ("recall", "dedupe", "topic", "confidence")

    for turn_id in sorted(actual_turns):
        actual_turn = actual_turns[turn_id]
        reference_turn = reference_turns[turn_id]
        exceeded_for_turn: list[dict[str, Any]] = []

        actual_recall = _maps_by_evidence(actual_turn, "recall")
        reference_recall = _maps_by_evidence(reference_turn, "recall")
        for evidence_id in sorted(set(actual_recall) | set(reference_recall)):
            entry = _numeric_entry(
                stage="recall",
                turn_id=turn_id,
                evidence_id=evidence_id,
                actual_units=actual_recall.get(evidence_id, {}).get(
                    "authority_score_units"
                ),
                reference_units=reference_recall.get(evidence_id, {}).get(
                    "authority_score_units"
                ),
                actual_scale=actual_scale,
                reference_scale=reference_scale,
                tolerance=tolerance,
            )
            stage_entries["recall"].append(entry)
            if entry["exceeds_tolerance"]:
                exceeded_for_turn.append(entry)

        actual_dedupe = _maps_by_evidence(actual_turn, "dedupe")
        reference_dedupe = _maps_by_evidence(reference_turn, "dedupe")
        for evidence_id in sorted(set(actual_dedupe) | set(reference_dedupe)):
            entry = _numeric_entry(
                stage="dedupe",
                turn_id=turn_id,
                evidence_id=evidence_id,
                actual_units=actual_dedupe.get(evidence_id, {}).get(
                    "merged_score_units"
                ),
                reference_units=reference_dedupe.get(evidence_id, {}).get(
                    "merged_score_units"
                ),
                actual_scale=actual_scale,
                reference_scale=reference_scale,
                tolerance=tolerance,
            )
            entry["authority_changed"] = actual_dedupe.get(
                evidence_id, {}
            ).get("authority_channel") != reference_dedupe.get(
                evidence_id, {}
            ).get("authority_channel")
            stage_entries["dedupe"].append(entry)
            if entry["exceeds_tolerance"] or entry["authority_changed"]:
                exceeded_for_turn.append(entry)

        actual_attr = {
            item["evidence_id"]: item
            for item in actual["evidence_attribution"]
        }
        reference_attr = {
            item["evidence_id"]: item
            for item in reference["evidence_attribution"]
        }
        actual_citation_ids = {
            item["evidence_id"]
            for item in actual_turn["dedupe"]["evidence"]
        }
        reference_citation_ids = {
            item["evidence_id"]
            for item in reference_turn["dedupe"]["evidence"]
        }
        for evidence_id in sorted(actual_citation_ids | reference_citation_ids):
            actual_topics = actual_attr.get(evidence_id, {}).get("topic_ids", [])
            reference_topics = reference_attr.get(evidence_id, {}).get(
                "topic_ids", []
            )
            membership_match = actual_topics == reference_topics
            entry = {
                "stage": "topic",
                "turn_id": turn_id,
                "evidence_id": evidence_id,
                "actual_topic_ids": actual_topics,
                "reference_topic_ids": reference_topics,
                "membership_delta": 0 if membership_match else 1,
                "exceeds_tolerance": not membership_match,
            }
            stage_entries["topic"].append(entry)
            if entry["exceeds_tolerance"]:
                exceeded_for_turn.append(entry)

        actual_conf = _maps_by_evidence(actual_turn, "confidence")
        reference_conf = _maps_by_evidence(reference_turn, "confidence")
        for evidence_id in sorted(set(actual_conf) | set(reference_conf)):
            entry = _numeric_entry(
                stage="confidence",
                turn_id=turn_id,
                evidence_id=evidence_id,
                actual_units=actual_conf.get(evidence_id, {}).get(
                    "contribution_units"
                ),
                reference_units=reference_conf.get(evidence_id, {}).get(
                    "contribution_units"
                ),
                actual_scale=actual_scale,
                reference_scale=reference_scale,
                tolerance=tolerance,
            )
            stage_entries["confidence"].append(entry)
            if entry["exceeds_tolerance"]:
                exceeded_for_turn.append(entry)
        total_entry = _numeric_entry(
            stage="confidence",
            turn_id=turn_id,
            evidence_id="__TOTAL__",
            actual_units=actual_turn["confidence"].get("confidence_units"),
            reference_units=reference_turn["confidence"].get(
                "confidence_units"
            ),
            actual_scale=actual_scale,
            reference_scale=reference_scale,
            tolerance=tolerance,
        )
        stage_entries["confidence"].append(total_entry)
        if total_entry["exceeds_tolerance"]:
            exceeded_for_turn.append(total_entry)

        if exceeded_for_turn:
            first = min(
                exceeded_for_turn,
                key=lambda item: (
                    stage_order.index(item["stage"]),
                    item["evidence_id"],
                ),
            )
            first_exceed_by_turn[str(turn_id)] = {
                "stage": first["stage"],
                "evidence_id": first["evidence_id"],
                "delta": first.get("delta"),
                "actual_topic_ids": first.get("actual_topic_ids"),
                "reference_topic_ids": first.get("reference_topic_ids"),
                "authority_changed": first.get("authority_changed", False),
            }

        actual_conclusion = actual_turn["confidence"].get("conclusion_bool")
        reference_conclusion = reference_turn["confidence"].get(
            "conclusion_bool"
        )
        if actual_conclusion != reference_conclusion:
            authority_changed = _authority_changed(
                actual_turn, reference_turn
            )
            cross_topic_changed = _cross_topic_changed(
                actual_turn, reference_turn
            )
            cross_delta = _cross_topic_contribution_delta(
                actual_turn,
                reference_turn,
                actual_scale,
                reference_scale,
            )
            actual_confidence = _confidence_value(actual_turn)
            reference_confidence = _confidence_value(reference_turn)
            direction = (
                "target_higher"
                if (actual_confidence or Decimal(0))
                > (reference_confidence or Decimal(0))
                else "target_lower"
            )
            causes: list[dict[str, Any]] = []
            if authority_changed:
                causes.append(
                    {
                        "cause": "dedupe_merge_order",
                        "detected": True,
                        "evidence_ids": authority_changed,
                    }
                )
            if cross_topic_changed or cross_delta > tolerance:
                causes.append(
                    {
                        "cause": "cross_topic_penalty",
                        "detected": True,
                        "evidence_ids": sorted(
                            set(cross_topic_changed)
                            | {
                                evidence_id
                                for evidence_id, item in actual_conf.items()
                                if item.get("cross_topic")
                            }
                        ),
                        "absolute_contribution_delta": str(cross_delta),
                    }
                )
            causes.append(
                {
                    "cause": "rounding_direction",
                    "detected": True,
                    "direction": direction,
                }
            )
            flips.append(
                {
                    "turn_id": turn_id,
                    "actual_conclusion": actual_conclusion,
                    "reference_conclusion": reference_conclusion,
                    "actual_confidence": str(actual_confidence),
                    "reference_confidence": str(reference_confidence),
                    "actual_answer_threshold": actual_turn["confidence"].get(
                        "answer_threshold"
                    ),
                    "reference_answer_threshold": reference_turn[
                        "confidence"
                    ].get("answer_threshold"),
                    "primary_cause": causes[0]["cause"],
                    "causes": causes,
                }
            )

    summaries: dict[str, Any] = {}
    for stage in stage_order:
        entries = stage_entries[stage]
        summaries[stage] = {
            "entry_count": len(entries),
            "exceed_count": sum(
                1 for entry in entries if entry["exceeds_tolerance"]
            ),
            "max_abs_delta": str(
                max(
                    (
                        abs(Decimal(entry["delta"]))
                        for entry in entries
                        if entry.get("delta") is not None
                    ),
                    default=Decimal(0),
                )
            ),
        }
    return {
        "schema_version": 1,
        "session_id": session.session_id,
        "tolerance": str(tolerance),
        "actual_domain": actual["fixed_point"],
        "reference_domain": reference["fixed_point"],
        "stage_order": list(stage_order),
        "stage_summaries": summaries,
        "first_exceed_by_turn": first_exceed_by_turn,
        "flip_count": len(flips),
        "flips": flips,
        "stages": stage_entries,
        "actual_stage_versions": actual["stage_versions"],
        "reference_stage_versions": reference["stage_versions"],
    }

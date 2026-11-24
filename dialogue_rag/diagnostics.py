"""JSON conversion, error propagation, and cross-run flip attribution."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .hashing import canonical_dumps


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Fraction):
        return round(value.numerator / value.denominator, 12)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_dumps(to_jsonable(payload)) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    differences = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            differences.append(float(item.get("abs_difference", 0.0) or 0.0))
        except (TypeError, ValueError):
            differences.append(0.0)
    exceeded = [item for item in items if isinstance(item, dict) and item.get("exceeded")]
    return {
        "item_count": len(items),
        "exceeded_count": len(exceeded),
        "max_abs_difference": max(differences, default=0.0),
        "items": to_jsonable(items),
    }


def build_error_propagation(
    turn_keys: list[tuple[str, int, str]],
    recall_results: dict[tuple[str, str], dict],
    turn_results: dict[tuple[str, str], dict],
) -> dict[str, Any]:
    stages_order = ("recall_scoring", "dedup_merge", "topic_attribution", "confidence_synthesis")
    turns: list[dict[str, Any]] = []
    stage_totals = {stage: {"exceeded_turns": 0, "exceeded_items": 0, "max_abs_difference": 0.0} for stage in stages_order}

    for session_id, _seq, turn_id in turn_keys:
        if (session_id, turn_id) not in turn_results:
            continue
        recall = recall_results[(session_id, turn_id)]
        scoring = turn_results[(session_id, turn_id)]
        stage_items = {
            "recall_scoring": recall["recall_errors"],
            "dedup_merge": recall["dedup_errors"],
            "topic_attribution": scoring["topic_errors"],
            "confidence_synthesis": [
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "evidence_id": None,
                    "stage": "confidence_synthesis",
                    **scoring["confidence_error"],
                }
            ]
            + [
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "evidence_id": item["evidence_id"],
                    "stage": "confidence_contribution",
                    **item["contribution_error"],
                }
                for item in scoring["chain_details"]
            ],
        }
        summaries = {stage: _stage_summary(items) for stage, items in stage_items.items()}
        first_exceeded = None
        for stage in stages_order:
            if summaries[stage]["exceeded_count"]:
                first_exceeded = stage
                stage_totals[stage]["exceeded_turns"] += 1
                stage_totals[stage]["exceeded_items"] += summaries[stage]["exceeded_count"]
                stage_totals[stage]["max_abs_difference"] = max(
                    stage_totals[stage]["max_abs_difference"], summaries[stage]["max_abs_difference"]
                )
                break
        for stage in stages_order:
            if summaries[stage]["exceeded_count"]:
                stage_totals[stage]["exceeded_items"] += 0
        turns.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "seq": scoring["turn_seq"],
                "first_exceeded_stage": first_exceeded,
                "candidate_conclusion": scoring["conclusion"],
                "reference_conclusion": scoring["reference_conclusion"],
                "conclusion_flipped": scoring["conclusion_flipped_vs_reference"],
                "flip_reason": scoring["flip_reason"],
                "stages": summaries,
            }
        )

    return {
        "reference": "exact rational/high-width complete calculation",
        "stage_order": list(stages_order),
        "stage_totals": stage_totals,
        "turns": turns,
    }


def compare_turn_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Classify conclusion flips between two complete diagnostic reports."""
    baseline_turns = {(item["session_id"], item["turn_id"]): item for item in baseline.get("turn_results", [])}
    candidate_turns = {(item["session_id"], item["turn_id"]): item for item in candidate.get("turn_results", [])}
    recall_baseline = {(item["session_id"], item["turn_id"]): item for item in baseline.get("recall_report", {}).get("turns", [])}
    recall_candidate = {(item["session_id"], item["turn_id"]): item for item in candidate.get("recall_report", {}).get("turns", [])}
    flips = []
    for key in sorted(set(baseline_turns) & set(candidate_turns)):
        before = baseline_turns[key]
        after = candidate_turns[key]
        if before["conclusion"] == after["conclusion"]:
            continue
        before_chain = [(item["source_turn_id"], item["evidence_id"]) for item in before["evidence_chain"]]
        after_chain = [(item["source_turn_id"], item["evidence_id"]) for item in after["evidence_chain"]]
        before_details = {
            (item.get("source_turn_id"), item.get("evidence_id")): item for item in before.get("chain_details", [])
        }
        after_details = {
            (item.get("source_turn_id"), item.get("evidence_id")): item for item in after.get("chain_details", [])
        }
        attribution_changed = False
        for chain_key in sorted(set(before_details) & set(after_details)):
            old_item = before_details[chain_key]
            new_item = after_details[chain_key]
            if old_item.get("evidence_topic_ids") != new_item.get("evidence_topic_ids"):
                attribution_changed = True
            if old_item.get("topic_penalty_reasons") != new_item.get("topic_penalty_reasons"):
                attribution_changed = True
        recall_before = recall_baseline.get(key, {})
        recall_after = recall_candidate.get(key, {})
        recall_changed = recall_before.get("output_fingerprint") != recall_after.get("output_fingerprint")
        dedup_changed = before_chain != after_chain or recall_changed
        topic_changed = before.get("topic_id") != after.get("topic_id") or attribution_changed
        reference_consistent = before.get("reference_conclusion") == after.get("reference_conclusion")
        if dedup_changed:
            classification = "dedup_or_recall_changed"
            criterion = "chain membership/order or dedup fingerprint differs before confidence rounding"
        elif topic_changed:
            classification = "topic_attribution_changed"
            criterion = "turn topic assignment differs while dedup chain is unchanged"
        elif reference_consistent:
            classification = "quantization_rounding"
            criterion = "exact reference conclusions agree; fixed-point conclusions differ"
        else:
            classification = "underlying_real_score_changed"
            criterion = "exact reference conclusions also differ"
        flips.append(
            {
                "session_id": key[0],
                "turn_id": key[1],
                "classification": classification,
                "criterion": criterion,
                "before_conclusion": before["conclusion"],
                "after_conclusion": after["conclusion"],
                "before_confidence": before["confidence"],
                "after_confidence": after["confidence"],
                "before_reference_confidence": before["reference_confidence"],
                "after_reference_confidence": after["reference_confidence"],
                "threshold": after["threshold"],
                "before_margin": before["margins"]["candidate_margin"],
                "after_margin": after["margins"]["candidate_margin"],
                "before_chain": before_chain,
                "after_chain": after_chain,
                "recall_changed": recall_changed,
                "dedup_changed": dedup_changed,
                "topic_changed": topic_changed,
                "attribution_changed": attribution_changed,
            }
        )
    return {"flip_count": len(flips), "flips": flips}

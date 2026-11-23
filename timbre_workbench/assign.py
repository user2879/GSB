"""Deterministic adjacent-segment speaker assignment and annotation checks."""

from __future__ import annotations

import math
from typing import Any

from .errors import E_ASSIGNMENT_CONFLICT
from .util import stable_float


def similarity(left: list[float], right: list[float], config: dict[str, Any]) -> float:
    if len(left) != len(right):
        raise ValueError("description vector dimensions differ")
    metric = config.get("metric", "rbf")
    if not left:
        return 0.0
    if metric == "rbf":
        squared = 0.0
        for a, b in zip(left, right):
            delta = float(a) - float(b)
            squared += delta * delta
        value = math.exp(-float(config.get("rbf_gamma", 8.0)) * (squared / len(left)))
    elif metric == "cosine":
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        value = dot / (left_norm * right_norm) if left_norm > 0 and right_norm > 0 else 0.0
    elif metric == "normalized_l1":
        value = 1.0 - sum(abs(float(a) - float(b)) for a, b in zip(left, right)) / len(left)
    else:
        raise ValueError(f"unsupported similarity metric: {metric}")
    return stable_float(max(0.0, min(1.0, value)))


def pair_decision(similarity_value: float, threshold: float) -> dict[str, Any]:
    similarity_value = stable_float(similarity_value)
    threshold = stable_float(threshold)
    if similarity_value > threshold:
        predicted_same = True
        reason = "similarity_above_threshold"
        tie = False
    elif similarity_value < threshold:
        predicted_same = False
        reason = "similarity_below_threshold"
        tie = False
    else:
        predicted_same = None
        reason = "threshold_equality_speaker_id_tiebreak"
        tie = True
    return {
        "similarity": similarity_value,
        "threshold": threshold,
        "predicted_same": predicted_same,
        "tie_at_threshold": tie,
        "decision_reason": reason,
    }


def _safe_suffix(file_name: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in file_name)
    return cleaned.strip("_") or "file"


def fold_assignment(
    segment_rows: list[dict[str, Any]],
    pair_decisions: dict[str, dict[str, Any]],
    config: dict[str, Any],
    conflict_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fold stable pair decisions into speaker groups, preserving order."""
    threshold = stable_float(float(config["similarity_threshold"]))
    tie_rule = config.get("tie_break", "speaker_id_lexicographic")
    assignments: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    previous_by_file: dict[str, dict[str, Any]] = {}
    group_counts: dict[tuple[str, str], int] = {}

    for row in segment_rows:
        file_name = row["file"]
        annotation = str(row["annotation"])
        segment_id = row["segment_id"]
        previous = previous_by_file.get(file_name)
        if previous is None:
            speaker_id = annotation
            group_counts[(file_name, annotation)] = group_counts.get((file_name, annotation), 0) + 1
            assignment = {
                "speaker_id": speaker_id,
                "speaker_annotation": annotation,
                "similarity_to_previous": None,
                "decision": "file_start",
                "decision_reason": "first_segment_in_file",
                "decision_rule": tie_rule,
                "annotation_consistent": True,
                "suppressed_candidates": [],
            }
        else:
            decision = pair_decisions[row["pair_decision_key"]]
            similarity_value = float(decision["similarity"])
            annotated_same = previous["annotation"] == annotation
            prev_speaker = str(previous["assignment"]["speaker_id"])
            if decision["tie_at_threshold"]:
                ordered = sorted([prev_speaker, annotation], key=lambda value: (value.casefold(), value))
                predicted_same = prev_speaker.casefold() <= annotation.casefold()
                if prev_speaker.casefold() == annotation.casefold():
                    predicted_same = prev_speaker <= annotation
                reason = "threshold_equality_speaker_id_lexicographic"
            else:
                predicted_same = bool(decision["predicted_same"])
                reason = str(decision["decision_reason"])

            if predicted_same:
                speaker_id = prev_speaker
                decision_name = "join_previous_speaker"
                new_candidate = annotation
                candidates = [
                    {"speaker_id": prev_speaker, "score": similarity_value, "accepted": True},
                ]
                if new_candidate != prev_speaker:
                    candidates.append(
                        {
                            "speaker_id": new_candidate,
                            "score": stable_float(1.0 - similarity_value),
                            "accepted": False,
                        }
                    )
            else:
                count = group_counts.get((file_name, annotation), 0) + 1
                group_counts[(file_name, annotation)] = count
                if count == 1:
                    speaker_id = annotation
                else:
                    speaker_id = f"{annotation}#{_safe_suffix(file_name)}#{count}"
                decision_name = "start_new_speaker"
                candidates = [
                    {"speaker_id": prev_speaker, "score": similarity_value, "accepted": False},
                    {
                        "speaker_id": speaker_id,
                        "score": stable_float(1.0 - similarity_value),
                        "accepted": True,
                    },
                ]
            suppressed = [candidate for candidate in candidates if not candidate["accepted"]]
            suppressed.sort(key=lambda candidate: (candidate["speaker_id"].casefold(), candidate["speaker_id"]))
            assignment = {
                "speaker_id": speaker_id,
                "speaker_annotation": annotation,
                "similarity_to_previous": similarity_value,
                "decision": decision_name,
                "decision_reason": reason,
                "decision_rule": tie_rule,
                "annotation_consistent": predicted_same == annotated_same,
                "suppressed_candidates": suppressed,
            }
            if predicted_same != annotated_same:
                conflicts.append(
                    {
                        "error_code": E_ASSIGNMENT_CONFLICT,
                        "file": file_name,
                        "previous_segment_id": previous["segment_id"],
                        "segment_id": segment_id,
                        "previous_annotation": previous["annotation"],
                        "current_annotation": annotation,
                        "similarity": similarity_value,
                        "threshold": threshold,
                        "predicted_same": predicted_same,
                        "annotated_same": annotated_same,
                        "predicted_speaker_id": speaker_id,
                        "resolution": "keep_prediction_record_conflict",
                        "sort_key": f"{file_name}|{previous['segment_index']:08d}|{row['segment_index']:08d}|{E_ASSIGNMENT_CONFLICT}",
                    }
                )
        assignments.append(assignment)
        previous_by_file[file_name] = {
            "file": file_name,
            "segment_id": segment_id,
            "segment_index": row["segment_index"],
            "annotation": annotation,
            "assignment": assignment,
        }
    conflicts.sort(key=lambda row: row["sort_key"])
    return assignments, conflicts

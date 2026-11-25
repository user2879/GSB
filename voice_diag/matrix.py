"""Confusion-matrix materialization and conclusion-count invariants."""

from __future__ import annotations

from typing import Any

from .fingerprint import sha256_json

REJECTED = "<REJECTED>"


def build_confusion_matrix(
    config: Any,
    conclusions: list[dict[str, Any]],
    high_references: dict[str, dict[str, Any]],
    *,
    expected_clip_count: int,
    cluster_assignments: dict[str, str] | None = None,
) -> dict[str, Any]:
    known_words = sorted(word["command_word"] for word in config.words)
    truth_set = set(known_words)
    truth_set.update(item["true_command"] for item in conclusions)
    truth_order = sorted(truth_set, key=lambda value: (value == "NEGATIVE", value))
    conclusion_order = known_words + [REJECTED]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {
        (truth, conclusion): [] for truth in truth_order for conclusion in conclusion_order
    }
    for result in conclusions:
        key = (result["true_command"], result["decision"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    assignment_by_word = cluster_assignments or {}
    cells: list[dict[str, Any]] = []
    for truth, conclusion in sorted(grouped, key=lambda item: (item[0] == "NEGATIVE", item[0], item[1] == REJECTED, item[1])):
        support = sorted(grouped[(truth, conclusion)], key=lambda item: item["clip_id"])
        boundary_ids = [item["clip_id"] for item in support if item["boundary"]]
        rejected_ids = [item["clip_id"] for item in support if item["rejected"]]
        high_entries = []
        for clip_id in boundary_ids:
            reference = high_references.get(clip_id)
            if reference is not None:
                high_entries.append(
                    {
                        "clip_id": clip_id,
                        "reference_conclusion": reference["reference_conclusion"],
                        "difference_direction": reference["difference_direction"],
                        "conclusion_changed": reference["conclusion_changed"],
                        "reference_top_similarity_permille": reference["reference_top_similarity_permille"],
                        "source_top_similarity_permille": reference["source_top_similarity_permille"],
                    }
                )
        reference_conclusion_counts: dict[str, int] = {}
        direction_counts: dict[str, int] = {}
        for entry in high_entries:
            reference_conclusion_counts[entry["reference_conclusion"]] = (
                reference_conclusion_counts.get(entry["reference_conclusion"], 0) + 1
            )
            direction_counts[entry["difference_direction"]] = direction_counts.get(entry["difference_direction"], 0) + 1
        cell = {
            "true_command": truth,
            "conclusion": conclusion,
            "predicted_cluster_id": None
            if conclusion == REJECTED
            else assignment_by_word.get(conclusion, conclusion),
            "support_fragment_count": len(support),
            "confusion_boundary_fragment_count": len(boundary_ids),
            "rejected_fragment_count": len(rejected_ids),
            "support_clip_ids": [item["clip_id"] for item in support],
            "confusion_boundary_clip_ids": boundary_ids,
            "rejected_clip_ids": rejected_ids,
            "truncation_boundary": bool(boundary_ids),
            "high_reference": (
                {
                    "reference_profile": config.reference_profile,
                    "comparisons": high_entries,
                    "reference_conclusion_counts": [
                        {"conclusion": key, "count": reference_conclusion_counts[key]}
                        for key in sorted(reference_conclusion_counts)
                    ],
                    "difference_direction_counts": [
                        {"direction": key, "count": direction_counts[key]} for key in sorted(direction_counts)
                    ],
                    "conclusion_changed_count": sum(1 for entry in high_entries if entry["conclusion_changed"]),
                }
                if high_entries
                else None
            ),
            "references": [item["clip_id"] for item in support],
            "result_cache_keys": [item["cache_key"] for item in support],
        }
        cell["cache_key"] = "matrix-cell:" + sha256_json(
            {
                "truth": truth,
                "conclusion": conclusion,
                "predicted_cluster_id": cell["predicted_cluster_id"],
                "profile": config.profile,
                "reference_profile": config.reference_profile,
                "results": cell["result_cache_keys"],
                "high_references": high_entries,
            }
        )
        cells.append(cell)

    support_total = sum(cell["support_fragment_count"] for cell in cells)
    boundary_total = sum(cell["confusion_boundary_fragment_count"] for cell in cells)
    rejected_total = sum(cell["rejected_fragment_count"] for cell in cells)
    conclusion_boundary_total = sum(1 for item in conclusions if item["boundary"])
    conclusion_rejected_total = sum(1 for item in conclusions if item["rejected"])
    unavailable_count = max(0, expected_clip_count - len(conclusions))
    consistent = (
        support_total == len(conclusions)
        and boundary_total == conclusion_boundary_total
        and rejected_total == conclusion_rejected_total
        and support_total + unavailable_count == expected_clip_count
    )
    result = {
        "rows": truth_order,
        "columns": conclusion_order,
        "cells": cells,
        "totals": {
            "expected_clip_count": expected_clip_count,
            "concluded_clip_count": len(conclusions),
            "unavailable_clip_count": unavailable_count,
            "support_fragment_count": support_total,
            "confusion_boundary_fragment_count": boundary_total,
            "rejected_fragment_count": rejected_total,
            "accepted_fragment_count": support_total - rejected_total,
        },
        "consistency": {
            "consistent": consistent,
            "support_equals_conclusion_list": support_total == len(conclusions),
            "boundary_equals_conclusion_list": boundary_total == conclusion_boundary_total,
            "rejected_equals_conclusion_list": rejected_total == conclusion_rejected_total,
            "support_plus_unavailable_equals_corpus": support_total + unavailable_count == expected_clip_count,
        },
    }
    result["cache_key"] = "confusion-matrix:" + sha256_json(
        {
            "cells": [cell["cache_key"] for cell in cells],
            "expected_clip_count": expected_clip_count,
            "profile": config.profile,
            "reject_threshold": config.reject_threshold,
        }
    )
    return result

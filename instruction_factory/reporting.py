from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import atomic_write_json, canonical_json, fingerprint_object


EQUIVALENCE_FIELDS = (
    "effective_config",
    "scoring_parameters",
    "configuration_hash",
    "seed_file_hash",
    "lexicon_hash",
    "quantization",
    "task_statistics",
    "planned_candidates",
    "candidates",
    "retained_candidate_ids",
    "quality_rejections",
    "boundary_samples",
    "near_duplicate_groups",
    "removed_duplicates",
    "scheduler",
    "failure_codes",
    "failures",
    "stage_versions",
)


def equivalence_projection(report: dict[str, Any]) -> dict[str, Any]:
    return {field: report.get(field) for field in EQUIVALENCE_FIELDS}


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    projection_left = equivalence_projection(left)
    projection_right = equivalence_projection(right)
    mismatches: list[str] = []
    for field in EQUIVALENCE_FIELDS:
        if canonical_json(projection_left[field]) != canonical_json(projection_right[field]):
            mismatches.append(field)
    return {
        "equal": not mismatches,
        "mismatches": mismatches,
        "left_projection_hash": fingerprint_object(projection_left),
        "right_projection_hash": fingerprint_object(projection_right),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    atomic_write_json(target, report)
    return target


def add_report_aliases(report: dict[str, Any]) -> dict[str, Any]:
    report["budget_ledger"] = report.get("scheduler", {}).get("ledger", [])
    report["failure_business_codes"] = [item["code"] for item in report.get("failures", [])]
    return report


def failure_report(
    *,
    mode: str,
    inputs: Any,
    failures: list[Any],
) -> dict[str, Any]:
    failure_dicts = [item.to_dict() if hasattr(item, "to_dict") else item for item in failures]
    return {
        "schema_version": 1,
        "pipeline": "deterministic-canonical-v1",
        "run_mode": mode,
        "effective_config": None,
        "task_statistics": {},
        "planned_candidates": [],
        "candidates": [],
        "retained_candidate_ids": [],
        "quality_rejections": [],
        "boundary_samples": [],
        "near_duplicate_groups": [],
        "removed_duplicates": [],
        "scheduler": {
            "ledger": [],
            "pending_queue": [],
        },
        "budget_ledger": [],
        "invalidation": [],
        "failure_codes": [item["code"] for item in failure_dicts],
        "failure_business_codes": [item["code"] for item in failure_dicts],
        "failures": failure_dicts,
        "stage_versions": {},
        "snapshot": None,
        "input_fingerprints": getattr(inputs, "fingerprints", {}),
    }


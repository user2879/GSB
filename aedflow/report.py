"""Fixed-order JSON diagnostic reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import BUSINESS_CODES


def build_report(context: dict[str, Any]) -> dict[str, Any]:
    boundary_rows = [decision.to_dict() for decision in context["boundary_decisions"]]
    trajectory_rows = [trajectory.to_dict() for trajectory in context["trajectories"]]
    cross_boundary = [row for row in trajectory_rows if row["crossed_boundaries"]]
    code_counts: dict[str, int] = {}
    for failure in context["failures"]:
        code_counts[failure["code"]] = code_counts.get(failure["code"], 0) + 1
    failures_sorted = sorted(
        context["failures"],
        key=lambda item: (item["stage"], item["subject_id"], item["code"], item["detail"]),
    )
    report = {
        "report_schema_version": "1.0",
        "run_id": context["run_id"],
        "sample_id": context["sample_id"],
        "config_fingerprint": context["config_fingerprint"],
        "header_version_set": list(context["header_version_set"]),
        "stage_status": context["stage_status"],
        "business_code_catalogue": {key: BUSINESS_CODES[key] for key in sorted(BUSINESS_CODES)},
        "business_code_counts": dict(sorted(code_counts.items())),
        "boundary_decisions": boundary_rows,
        "frame_decision_distribution": dict(sorted(context["frame_distribution"].items())),
        "trajectory_counts_by_class": dict(sorted(context["trajectory_counts"].items())),
        "trajectories": trajectory_rows,
        "cross_boundary_trajectories": cross_boundary,
        "smoothing_parameters": context["smoothing_parameters"],
        "compatibility_records": context["compat_records"],
        "resampling_residuals": [
            record
            for record in context["compat_records"]
            if record["resampled"]
        ],
        "alignment_residuals": context["alignment_residuals"],
        "evaluation": context["evaluation"],
        "dropped_short_trajectories": context["dropped_short"],
        "duplicate_resolutions": context["duplicates"],
        "failures": failures_sorted,
        "exit": context["exit"],
    }
    return report


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from scorecard_app.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def base_inputs(rows: int = 480) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = load_yaml(ROOT / "configs" / "experiment.yaml")
    schema = load_yaml(ROOT / "configs" / "synthetic.yaml")
    schema["rows"] = rows
    return experiment, schema


def clone(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)


CONTENT_COLUMN_FIELDS = (
    "name",
    "type",
    "direction",
    "status",
    "bin_count",
    "nonmissing_bin_count",
    "missing_bin",
    "iv",
    "smoothed_bin_count",
    "monotonic_correction_count",
    "corrections",
    "bins",
    "category_to_group",
)


def content_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "columns": [
            {field: column[field] for field in CONTENT_COLUMN_FIELDS}
            for column in report.get("columns", [])
        ],
        "evaluation": report.get("evaluation"),
        "failures": report.get("failures"),
    }

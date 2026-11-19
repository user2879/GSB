from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from bucketpack.spec import Spec, load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def sample_spec() -> Spec:
    return load_spec(ROOT / "samples" / "spec.json")


def sample_raw() -> dict[str, Any]:
    return json.loads((ROOT / "samples" / "spec.json").read_text(encoding="utf-8"))


def tiny_raw() -> dict[str, Any]:
    return {
        "format_version": 1,
        "sources": [
            {
                "id": "s",
                "seed": 9,
                "rows": 4,
                "primary_key": "id",
                "columns": [
                    {"name": "id", "type": "int", "min": 1, "max": 1000},
                    {"name": "v", "type": "int", "min": 0, "max": 9},
                ],
                "null_bitmaps": {"v": "0100"},
            }
        ],
        "rules": [
            {
                "id": "vb",
                "source": "s",
                "input": {"kind": "column", "name": "v"},
                "boundaries": [3, 7],
                "scale": 0,
            }
        ],
        "statistics": [{"id": "stat", "source": "s", "refs": ["vb"], "metric": "sum_counts"}],
        "packing": {
            "rows_per_pack": 2,
            "pack_width_budget": 20,
            "global_width_budget": 20,
            "pack_quota_per_source": 1,
            "workers": 2,
        },
        "accounting": {"unit_cost_micros": 3, "budget_micros": 10000},
    }


def tiny_spec(raw: dict[str, Any] | None = None, base_dir: str = "") -> Spec:
    return validate_spec(copy.deepcopy(raw or tiny_raw()), base_dir=base_dir)


def write_known_data_spec(tmp_path: Path) -> Spec:
    data = {
        "format_version": 1,
        "source_id": "known",
        "rows": [
            {"id": 1, "v": 0.0},
            {"id": 2, "v": 9.99},
            {"id": 3, "v": 10.0},
            {"id": 4, "v": 19.99},
        ],
    }
    (tmp_path / "known.json").write_text(json.dumps(data), encoding="utf-8")
    raw = {
        "format_version": 1,
        "sources": [
            {
                "id": "known",
                "seed": 1,
                "rows": 4,
                "primary_key": "id",
                "data_file": "known.json",
                "columns": [
                    {"name": "id", "type": "int", "min": 1, "max": 9},
                    {"name": "v", "type": "float", "min": 0.0, "max": 20.0, "precision": 2, "null_fill": 5.0},
                ],
                "null_bitmaps": {"v": "0000"},
            }
        ],
        "rules": [
            {
                "id": "known.vb",
                "source": "known",
                "input": {"kind": "column", "name": "v"},
                "boundaries": [0, 1000, 1500],
                "scale": 2,
            },
            {
                "id": "known.vb2",
                "source": "known",
                "input": {"kind": "bucket", "name": "known.vb"},
                "boundaries": [1, 2],
                "scale": 0,
            },
        ],
        "statistics": [],
        "packing": {
            "rows_per_pack": 3,
            "pack_width_budget": 100,
            "global_width_budget": 100,
            "pack_quota_per_source": 1,
            "workers": 2,
        },
        "accounting": {"unit_cost_micros": 1, "budget_micros": 1000},
    }
    return validate_spec(raw, base_dir=str(tmp_path))

"""Deterministic fixture generator (samples + pipeline specification).

The data is hard-coded rather than random so every machine and every run has
byte-identical inputs.  Integer and float columns plus explicit nulls exercise
every pipeline stage.
"""

from __future__ import annotations

import json
import os
from typing import Any

COLUMNS = {
    "visits": {"type": "int"},
    "score": {"type": "float"},
    "label": {"type": "int"},
}

TRAIN_A = [
    {"visits": 12, "score": 1.25, "label": 0},
    {"visits": 7, "score": 2.5, "label": 1},
    {"visits": None, "score": 3.75, "label": 0},
    {"visits": 21, "score": None, "label": 1},
    {"visits": 5, "score": 0.5, "label": 0},
    {"visits": 16, "score": 4.25, "label": 1},
    {"visits": 9, "score": 2.0, "label": 0},
    {"visits": 3, "score": 1.75, "label": 1},
]

TRAIN_B = [
    {"visits": 11, "score": 2.25, "label": 0},
    {"visits": 6, "score": 1.5, "label": 0},
    {"visits": 14, "score": 3.0, "label": 1},
    {"visits": 8, "score": 2.75, "label": 0},
    {"visits": None, "score": 0.0, "label": 1},
]


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_pipeline(normalize_scale: int = 3,
                   target_rows: int = 4) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sources": [
            {"id": "train_a", "path": os.path.join("samples", "train_a.json"),
             "columns": dict(COLUMNS)},
            {"id": "train_b", "path": os.path.join("samples", "train_b.json"),
             "columns": dict(COLUMNS)},
        ],
        "stages": [
            {"id": "read", "type": "read", "version": 1, "upstream": [],
             "params": {"chunk_rows": 4,
                        "source_ids": ["train_a", "train_b"]}},
            {"id": "types", "type": "normalize_types", "version": 1,
             "upstream": ["read"],
             "params": {"fixed_scale": 1000, "rounding": "toward_zero"}},
            {"id": "fill", "type": "fill_missing", "version": 1,
             "upstream": ["types"], "params": {"fill_value": 0}},
            {"id": "scale", "type": "normalize", "version": 1,
             "upstream": ["fill"],
             "params": {"scale": normalize_scale, "divisor": 1,
                        "rounding": "toward_zero"}},
            {"id": "clip", "type": "clip", "version": 1,
             "upstream": ["scale"],
             "params": {"lower": -20000, "upper": 40000,
                        "max_outlier_bp": 2000}},
            {"id": "pack", "type": "shard", "version": 1,
             "upstream": ["clip"],
             "params": {"target_rows": target_rows,
                        "remainder_rule": "keep_partial"}},
        ],
    }


def generate_fixtures(base_dir: str, normalize_scale: int = 3,
                      target_rows: int = 4) -> str:
    sample_dir = os.path.join(base_dir, "samples")
    _write_json(os.path.join(sample_dir, "train_a.json"),
                {"rows": TRAIN_A})
    _write_json(os.path.join(sample_dir, "train_b.json"),
                {"rows": TRAIN_B})
    spec_path = os.path.join(base_dir, "pipeline.json")
    _write_json(spec_path, build_pipeline(normalize_scale, target_rows))
    return spec_path

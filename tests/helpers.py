from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8", newline="\n")


def base_config(**scheduling: Any) -> dict[str, Any]:
    return {
        "quantization": {
            "scheme": "asymmetric",
            "bit_width": 8,
            "group_granularity": 4,
            "percentile_lower_permille": 100,
            "percentile_upper_permille": 900,
            "fractional_bits": 8,
            "quantization_levels": 1,
            "scale_effective_digits": 15,
        },
        "layers": {},
        "distillation": {
            "temperature": 2.0,
            "steps": 3,
            "soft_label_weight": 0.5,
            "tolerance": 0.02,
        },
        "scheduling": {
            "thread_pool_workers": 2,
            "global_calibration_point_budget": 1000,
            "per_layer_calibration_quota": 1000,
            "batch_quantization_point_budget": 4,
            "snapshot_commit_interval_batches": 1,
            **scheduling,
        },
    }


def calibration_fixture() -> dict[str, Any]:
    return {
        "input_dim": 4,
        "output_dim": 2,
        "samples": [
            {"id": "s0", "input": [0.0, 0.5, -0.5, 1.0], "reference_output": [0.0, 0.1]},
            {"id": "s1", "input": [1.0, -1.0, 0.25, 0.0], "reference_output": [0.2, -0.1]},
        ],
    }

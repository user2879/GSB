from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qcolumn_diag.pipeline import run_workbench

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def base_config(**tensor_overrides: Any) -> dict[str, Any]:
    tensor = {
        "tensor_id": "t0",
        "shape": [4, 3],
        "column_axis": 1,
        "quantile_lower_permille": 100,
        "quantile_upper_permille": 900,
        "scheme": "asymmetric",
        "bit_width_candidates": [4, 8],
        "active_bit_width": 8,
        "fractional_bits": 6,
        "saturation_strategy": "saturate",
        "zero_scale_policy": "clamp_to_min_scale",
        "tolerance": 0.03,
        "group_over_tolerance_limit": 0.5,
        "column_groups": [
            {"group_id": "left", "columns": [0, 1]},
            {"group_id": "right", "columns": [2]},
        ],
    }
    global_keys = {
        "max_workers",
        "calibration_element_budget",
        "snapshot_commit_interval_batches",
        "allowed_bit_widths",
        "interpolation",
        "tie_break",
        "scale_quantization",
        "config_id",
    }
    globals_ = {
        key: tensor_overrides.pop(key)
        for key in list(tensor_overrides)
        if key in global_keys
    }
    tensor.update(tensor_overrides)
    config = {
        "schema_version": "1.0",
        "config_id": "test-config",
        "interpolation": "linear",
        "tie_break": "stable_ascending_first_occurrence",
        "scale_quantization": "round_half_away_from_zero",
        "allowed_bit_widths": [4, 6, 8],
        "max_workers": 2,
        "calibration_element_budget": 100,
        "snapshot_commit_interval_batches": 2,
        "tensors": [tensor],
    }
    config.update(globals_)
    return config


def base_sample(**tensor_overrides: Any) -> dict[str, Any]:
    tensor = {
        "dtype": "float64",
        "shape": [4, 3],
        "generator": {"kind": "normal", "seed": 1234, "loc": 0.0, "scale": 1.0},
        "column_offsets": [-2.0, 0.0, 2.0],
        "column_scales": [1.0, 1.2, 0.8],
        "missing_values": [[1, 2]],
        "injections": [
            {"id": "I-1", "row": 0, "column": 0, "operation": "set", "value": 5.0},
            {"id": "I-2", "row": 2, "column": 1, "operation": "set", "value": 0.0},
            {"id": "I-3", "row": 3, "column": 2, "operation": "set", "value": -5.0},
        ],
    }
    tensor.update(tensor_overrides)
    return {
        "schema_version": "1.0",
        "sample_id": "test-sample",
        "default_dtype": "float64",
        "tensors": {"t0": tensor},
    }


def run_in_dir(
    root: Path,
    config: dict[str, Any],
    sample: dict[str, Any],
    *,
    name: str,
    full_rebuild: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    config_path = write_json(root / f"{name}.config.json", config)
    sample_path = write_json(root / f"{name}.sample.json", sample)
    state_dir = root / f"{name}.state"
    report_path = root / f"{name}.report.json"
    report = run_workbench(
        config_path=config_path,
        sample_path=sample_path,
        state_dir=state_dir,
        report_path=report_path,
        full_rebuild=full_rebuild,
    )
    return report, config_path, sample_path


DETERMINISTIC_FIELDS = (
    "metadata",
    "configuration",
    "sample_metadata",
    "injection_hits",
    "columns",
    "sensitivity",
    "failures",
    "summary",
)


def assert_deterministic_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left["equivalence"]["deterministic_result_hash"] == right[
        "equivalence"
    ]["deterministic_result_hash"]
    for field in DETERMINISTIC_FIELDS:
        assert left[field] == right[field], field

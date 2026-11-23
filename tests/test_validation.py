from __future__ import annotations

import json

from dtw_diag.config import parse_config
from dtw_diag.errors import (
    E_BANDWIDTH,
    E_DUPLICATE_ID,
    E_EDIT_TYPE,
    E_EMPTY_SEQUENCE,
    E_MAX_PATH_LENGTH,
    E_OBS_VALUE,
    E_QUANT_LEVEL,
    E_SEGMENT_LENGTH,
    E_STEP_MODE,
)
from dtw_diag.storage import load_edit_plan, load_library

from conftest import make_config, write_jsonl


def _base_raw():
    return {
        "step_mode": "symmetric",
        "bandwidth_permille": 250,
        "distance_coefficient": 2,
        "fixed_point": {"fraction_bits": 8, "quantization_step": 4},
        "missing_marker": None,
        "missing_penalty": 9,
        "max_path_length": 50,
        "path_length_policy": "reject",
        "thread_pool_workers": 2,
        "cell_recompute_budget": 20,
        "snapshot_interval_batches": 1,
    }


def test_config_business_codes():
    raw = _base_raw()
    raw["step_mode"] = "zigzag"
    result = parse_config(raw)
    assert result.config is None
    assert E_STEP_MODE in {item.code for item in result.failures}

    raw = _base_raw()
    raw["bandwidth_permille"] = 1001
    result = parse_config(raw)
    assert result.config is None
    assert E_BANDWIDTH in {item.code for item in result.failures}

    raw = _base_raw()
    raw["fixed_point"]["quantization_step"] = 0
    result = parse_config(raw)
    assert result.config is None
    assert E_QUANT_LEVEL in {item.code for item in result.failures}

    raw = _base_raw()
    raw["max_path_length"] = 0
    result = parse_config(raw)
    assert result.config is None
    assert E_MAX_PATH_LENGTH in {item.code for item in result.failures}


def test_library_reports_invalid_rows_without_silent_skip(tmp_path):
    config = make_config()
    rows = [
        {"id": "A", "observations": [0.1, 0.2], "segment_ids": [0, 1]},
        {"id": "A", "observations": [0.3, 0.4]},
        {"id": "EMPTY", "observations": []},
        {"id": "NAN", "observations": [0.1, float("nan")]},
        {"id": "SEG", "observations": [0.1, 0.2], "segment_ids": [0]},
    ]
    path = tmp_path / "library.jsonl"
    write_jsonl(path, rows)
    result = load_library(path, config)
    codes = {item.code for item in result.failures}
    assert E_DUPLICATE_ID in codes
    assert E_EMPTY_SEQUENCE in codes
    assert E_OBS_VALUE in codes
    assert E_SEGMENT_LENGTH in codes
    assert set(result.library.records) == {"A"}
    assert len(result.library.invalid_entries) == 4


def test_edit_plan_shape_and_replace_alias(tmp_path):
    plan = {
        "batches": [
            {
                "batch": 1,
                "edits": [
                    {
                        "type": "sequence_replace",
                        "record": {
                            "id": "Q1",
                            "observations": [0.1, 0.2],
                            "segment_ids": [0, 0],
                        },
                    },
                    {"type": "mystery"},
                ],
            }
        ]
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    result = load_edit_plan(path)
    assert [item.code for item in result.failures] == [E_EDIT_TYPE]
    assert len(result.batches[0].edits) == 1

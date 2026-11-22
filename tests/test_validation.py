from __future__ import annotations

import copy
from pathlib import Path

import pytest

from qcolumn_diag.config import load_config
from qcolumn_diag.sample import load_sample
from tests.helpers import base_config, base_sample, write_json


@pytest.mark.parametrize(
    ("field", "value", "failure_field"),
    [
        ("shape", [0, 3], "shape[0]"),
        ("column_axis", 2, "column_axis"),
        ("quantile_lower_permille", 900, "quantile_lower_permille"),
        ("bit_width_candidates", [3], "bit_width_candidates[0]"),
        ("fractional_bits", 0, "fractional_bits"),
        ("saturation_strategy", "unknown", "saturation_strategy"),
        ("tolerance", float("inf"), "tolerance"),
    ],
)
def test_invalid_tensor_fields_are_reported(
    tmp_path: Path, field: str, value: object, failure_field: str
) -> None:
    config = base_config(**{field: value})
    path = write_json(tmp_path / "config.json", config)
    result = load_config(path)
    assert result.config is not None
    assert result.failures
    assert all(item["error_code"] == "E_CONFIG_INVALID" for item in result.failures)
    assert any(item["field"].endswith(failure_field) for item in result.failures)
    assert result.config.tensors == ()


def test_invalid_global_workers_and_bad_tensor_do_not_block_valid_tensor(
    tmp_path: Path,
) -> None:
    config = base_config()
    config["max_workers"] = 0
    valid = copy.deepcopy(config["tensors"][0])
    valid["tensor_id"] = "valid"
    invalid = copy.deepcopy(config["tensors"][0])
    invalid["tensor_id"] = "invalid"
    invalid["shape"] = [0, 3]
    config["tensors"] = [invalid, valid]
    path = write_json(tmp_path / "config.json", config)
    result = load_config(path)
    assert result.config is not None
    assert [tensor.tensor_id for tensor in result.config.tensors] == ["valid"]
    assert result.config.max_workers == 1
    codes = [item["error_code"] for item in result.failures]
    assert "E_CONFIG_INVALID" in codes
    assert any(item["field"] == "max_workers" for item in result.failures)
    assert any(item["field"].endswith("shape[0]") for item in result.failures)


def test_sample_shape_seed_and_coordinate_validation(tmp_path: Path) -> None:
    config_path = write_json(tmp_path / "config.json", base_config())
    config_result = load_config(config_path)
    assert config_result.config is not None
    tensor = config_result.config.tensors[0]

    bad_shape = base_sample(shape=[3, 3])
    result = load_sample(write_json(tmp_path / "bad_shape.json", bad_shape), [tensor])
    assert not result.tensors
    assert any(item["error_code"] == "E_SAMPLE_INVALID" for item in result.failures)

    bad_seed = base_sample()
    bad_seed["tensors"]["t0"]["generator"]["seed"] = -1
    result = load_sample(write_json(tmp_path / "bad_seed.json", bad_seed), [tensor])
    assert not result.tensors
    assert any("seed" in item["field"] for item in result.failures)

    bad_coordinate = base_sample(missing_values=[[99, 0]])
    result = load_sample(
        write_json(tmp_path / "bad_coordinate.json", bad_coordinate), [tensor]
    )
    assert not result.tensors
    assert any("missing_values" in item["field"] for item in result.failures)


def test_duplicate_group_assignment_is_rejected(tmp_path: Path) -> None:
    config = base_config(
        column_groups=[
            {"group_id": "a", "columns": [0, 1]},
            {"group_id": "b", "columns": [1, 2]},
        ]
    )
    result = load_config(write_json(tmp_path / "config.json", config))
    assert result.config is not None
    assert result.config.tensors == ()
    assert any("already assigned" in item["message"] for item in result.failures)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layerpack.errors import (
    E_CALIB_MISSING,
    E_CONFIG_INVALID,
    E_NONFINITE_WEIGHT,
    E_WEIGHT_SHAPE_MISMATCH,
)
from layerpack.loader import load_inputs

from conftest import clone, write_config, write_json


def _codes(state) -> set[str]:
    return {item.code for item in state.failures}


def test_valid_sample_bundle(sample_paths):
    state, config = load_inputs(
        sample_paths["weights"], sample_paths["calibration"], sample_paths["config"]
    )
    assert not state.failures
    assert sorted(state.layers) == [
        "embed",
        "encoder.attn.k",
        "encoder.attn.q",
        "encoder.attn.v",
        "encoder.norm",
        "head",
    ]
    assert config.concurrency == 4
    assert config.point_budget_per_batch == 65536


def test_shape_mismatch_code(copy_inputs):
    lines = copy_inputs["weights"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["shape"] = [3, 3]
    lines[0] = json.dumps(record)
    copy_inputs["weights"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    state, _ = load_inputs(
        copy_inputs["weights"], copy_inputs["calibration"], copy_inputs["config"]
    )
    assert E_WEIGHT_SHAPE_MISMATCH in _codes(state)


def test_duplicate_and_nonmonotonic_versions(copy_inputs):
    lines = copy_inputs["weights"].read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(lines[0])
    lines.append(json.dumps(duplicate))
    copy_inputs["weights"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    state, _ = load_inputs(
        copy_inputs["weights"], copy_inputs["calibration"], copy_inputs["config"]
    )
    assert E_WEIGHT_SHAPE_MISMATCH in _codes(state)

    lines = copy_inputs["weights"].read_text(encoding="utf-8").splitlines()[:-1]
    second = json.loads(lines[1])
    second["content_version"] = 0
    lines[1] = json.dumps(second)
    copy_inputs["weights"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    state, _ = load_inputs(
        copy_inputs["weights"], copy_inputs["calibration"], copy_inputs["config"]
    )
    assert E_WEIGHT_SHAPE_MISMATCH in _codes(state)


def test_nonfinite_weight_code(copy_inputs):
    lines = copy_inputs["weights"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["weights"][0] = float("nan")
    lines[0] = json.dumps(record)
    copy_inputs["weights"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    state, _ = load_inputs(
        copy_inputs["weights"], copy_inputs["calibration"], copy_inputs["config"]
    )
    assert E_NONFINITE_WEIGHT in _codes(state)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["quantization"].__setitem__("default_bits", 3),
        lambda c: c["runtime"].__setitem__("concurrency", 0),
        lambda c: c["runtime"].__setitem__("point_budget_per_batch", 0),
        lambda c: c.__setitem__("full_precision_layers", ["missing.layer"]),
        lambda c: c["layers"]["embed"].__setitem__("group_size", 3),
    ],
)
def test_config_invalid_codes(copy_inputs, base_config, mutation):
    config = clone(base_config)
    mutation(config)
    path = write_config(copy_inputs["config"], config)
    state, _ = load_inputs(
        copy_inputs["weights"], copy_inputs["calibration"], path
    )
    assert E_CONFIG_INVALID in _codes(state)


def test_calibration_missing_codes(copy_inputs, base_calibration):
    calibration = clone(base_calibration)
    calibration["samples"][0]["input"] = [1.0, 2.0]
    path = write_json(copy_inputs["calibration"], calibration)
    state, _ = load_inputs(copy_inputs["weights"], path, copy_inputs["config"])
    assert E_CALIB_MISSING in _codes(state)

    missing = copy_inputs["calibration"].with_name("missing.json")
    state, _ = load_inputs(copy_inputs["weights"], missing, copy_inputs["config"])
    assert E_CALIB_MISSING in _codes(state)

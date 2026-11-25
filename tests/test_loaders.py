import json

from qcalib.errors import E_INPUT_SCHEMA, FailureList
from qcalib.loaders import (
    load_calibration,
    load_command_stream,
    load_config,
    load_weight_package,
)
from qcalib.models import WeightLayer
import numpy as np

from conftest import ROOT  # noqa: E402


def test_load_deterministic_package_and_config():
    failures = FailureList()
    layers = load_weight_package(ROOT / "samples" / "weights.jsonl", failures)
    assert [layer.name for layer in layers] == [
        "embedding_proj", "retrieval_head", "fusion_block", "score_mixer"]
    assert [layer.content_version for layer in layers] == [1, 2, 3, 4]
    calibration = load_calibration(
        ROOT / "samples" / "calibration.json", layers, failures)
    assert calibration.dim == 8
    assert calibration.sample_count == 12
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    assert config.workers == 4
    assert config.resolve_layer("fusion_block").level == "int4"
    assert len(failures) == 0


def test_schema_failures_do_not_abort(tmp_path):
    bad_line = json.dumps({
        "layer": "bad", "shape": [2, 2],
        "weights": [1.0, float("nan"), 3.0, 4.0],
        "content_version": 1, "group_hint": 2})
    good_line = json.dumps({
        "layer": "good", "shape": [2], "weights": [0.1, 0.2],
        "content_version": 2, "group_hint": 2})
    package = tmp_path / "weights.jsonl"
    package.write_text(
        json.dumps({"layer": "x", "shape": [1], "weights": [1.0],
                    "content_version": 1, "group_hint": 1,
                    "extra": True}) + "\n"
        + bad_line + "\n" + good_line + "\n", encoding="utf-8")
    failures = FailureList()
    layers = load_weight_package(package, failures)
    assert [layer.name for layer in layers] == ["good"]
    codes = {failure.code for failure in failures}
    assert codes == {E_INPUT_SCHEMA}
    assert len(failures) == 2


def test_monotonic_version_and_group_divisibility(tmp_path):
    line = json.dumps({
        "layer": "v", "shape": [3], "weights": [1.0, 2.0, 3.0],
        "content_version": 1, "group_hint": 2})
    package = tmp_path / "weights.jsonl"
    package.write_text(line + "\n", encoding="utf-8")
    failures = FailureList()
    layers = load_weight_package(package, failures)
    assert layers == []
    assert failures.to_list()[0]["message"] == (
        "group_hint must divide layer element count")


def test_command_stream_rejects_unknown_op(tmp_path):
    stream = tmp_path / "commands.json"
    stream.write_text(json.dumps({"commands": [
        {"op": "noop"}, {"op": "explode"}]}), encoding="utf-8")
    failures = FailureList()
    commands = load_command_stream(stream, failures)
    assert [command["op"] for command in commands] == ["noop"]
    assert failures.to_list()[0]["code"] == E_INPUT_SCHEMA


def test_calibration_dim_mismatch_recorded():
    failures = FailureList()
    layer = WeightLayer("l", (2, 4), np.zeros((2, 4)), 1, 4)
    data = {
        "inputs": np.zeros((2, 8)).tolist(),
        "references": np.zeros((2, 2)).tolist(),
        "sample_ids": ["a", "b"],
        "tier_sizes": {"small": 1},
    }
    path = ROOT / "samples" / "_tmp_cal.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    try:
        cal = load_calibration(path, [layer], failures)
        assert cal.dim == 8
        assert failures.to_list()[0]["code"] == E_INPUT_SCHEMA
    finally:
        path.unlink()

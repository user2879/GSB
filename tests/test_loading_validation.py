from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_workbench.engine import WorkbenchEngine
from quant_workbench.errors import (
    E_CONFIG_INVALID,
    E_LAYER_INVALID,
    E_SHAPE_MISMATCH,
    E_VERSION_NOT_MONOTONIC,
    E_WEIGHT_NOT_FINITE,
    WorkbenchError,
)
from quant_workbench.loading import (
    load_calibration,
    load_commands,
    load_config,
    load_weight_pack,
)
from quant_workbench.models import Command

from .conftest import write_jsonl


def test_weight_pack_reports_structured_line_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad-weights.jsonl"
    write_jsonl(
        path,
        [
            {"layer": "a", "version": 1, "shape": [1, 2], "weights": [[1.0, 2.0]]},
            {"layer": "a", "version": 2, "shape": [1, 2], "weights": [[3.0, 4.0]]},
            {"layer": "b", "version": 3, "shape": [2, 2], "weights": [1.0, 2.0]},
            {"layer": "c", "version": 4, "shape": [1, 1], "weights": [[float("nan")]]},
        ],
    )
    layers, issues = load_weight_pack(path, command_index=7)
    assert set(layers) == {"a"}
    by_code = {issue.code for issue in issues}
    assert {E_LAYER_INVALID, E_SHAPE_MISMATCH, E_WEIGHT_NOT_FINITE} <= by_code
    assert all(issue.command_index == 7 for issue in issues)
    assert [issue.line_number for issue in issues] == [2, 3, 4]


def test_weight_pack_enforces_monotonic_versions(tmp_path: Path) -> None:
    path = tmp_path / "versions.jsonl"
    write_jsonl(
        path,
        [
            {"layer": "a", "version": 3, "shape": [1], "weights": [1.0]},
            {"layer": "b", "version": 2, "shape": [1], "weights": [2.0]},
        ],
    )
    layers, issues = load_weight_pack(path)
    assert set(layers) == {"a"}
    assert issues[0].code == E_VERSION_NOT_MONOTONIC
    assert issues[0].line_number == 2


@pytest.mark.parametrize(
    "config_text,needle",
    [
        ("allowed_bit_widths: [4, 8]\ndefaults:\n  bits: 3\n", "default bits"),
        ("allowed_bit_widths: [4, 8]\ndefaults:\n  group_size: 0\n", "group_size"),
        ("allowed_bit_widths: [4, 8]\ndefaults:\n  fractional_bits: 0\n", "fractional_bits"),
        ("latency_budget_ms: 0\n", "latency_budget_ms"),
        ("max_threads: 9\nthread_limit: 8\n", "exceeds configured upper limit"),
    ],
)
def test_config_validation(tmp_path: Path, config_text: str, needle: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(config_text, encoding="utf-8")
    with pytest.raises(WorkbenchError) as error:
        load_config(path)
    assert error.value.code == E_CONFIG_INVALID
    assert needle in str(error.value)


def test_calibration_and_commands_load(sample_paths: dict[str, Path]) -> None:
    calibration, issues = load_calibration(sample_paths["calibration"])
    assert len(calibration.samples) == 10
    assert not issues
    commands, command_issues = load_commands(sample_paths["commands"])
    assert not command_issues
    assert [command.op for command in commands] == [
        "set_bits",
        "set_group_size",
        "set_scheme",
        "replace_weight",
        "set_temperature",
        "replace_calibration",
    ]


def test_command_unknown_layer_and_bad_replacement(sample_paths: dict[str, Path]) -> None:
    config = load_config(sample_paths["config"])
    config.latency_budget_ms = 1000
    config.point_batch_budget = 1_000_000
    config.worker_delay_ms = 0
    weights, weight_issues = load_weight_pack(sample_paths["weights"])
    calibration, calibration_issues = load_calibration(sample_paths["calibration"])
    engine = WorkbenchEngine(
        config,
        weights,
        calibration,
        base_issues=weight_issues + calibration_issues,
        command_base_dir=sample_paths["commands"].parent,
    )
    try:
        responses = [
            engine.process_command(
                Command(1, 1, "set_bits", "missing", {"bits": 4}, {"cmd": "set_bits"})
            ),
            engine.process_command(
                Command(
                    2,
                    2,
                    "replace_weight",
                    "layer.0",
                    {"shape": [99, 99], "weights": [[1.0]], "version": 2},
                    {"cmd": "replace_weight"},
                )
            ),
            engine.process_command(
                Command(
                    3,
                    3,
                    "replace_weight",
                    "layer.0",
                    {
                        "shape": [6, 8],
                        "weights": [[float("nan")] * 8 for _ in range(6)],
                        "version": 2,
                    },
                    {"cmd": "replace_weight"},
                )
            ),
        ]
        assert responses[0]["is_latest"] is True
        codes = [issue["code"] for issue in engine.diagnostics()["failures"]]
        assert codes == [E_LAYER_INVALID, E_SHAPE_MISMATCH, E_WEIGHT_NOT_FINITE]
        failure_commands = [
            issue["command_index"] for issue in engine.diagnostics()["failures"]
        ]
        assert failure_commands == [1, 2, 3]
    finally:
        engine.close()


def test_nested_config_and_mapping_layer_overrides(tmp_path: Path) -> None:
    path = tmp_path / "nested.yaml"
    path.write_text(
        """
quantization:
  allowed_bit_widths: [4, 8]
  defaults:
    target_bits: 8
    grouping_granularity: 4
    fixed_fractional_bits: 3
    scheme: symmetric
  target_bits:
    layer.a: 4
    layer.b: 8
  group_sizes:
    layer.a: 2
  layers:
    layer.a:
      scheme: asymmetric
interaction:
  latency_budget_ms: 9
  thread_pool:
    max_threads: 2
    limit: 3
snapshot:
  version: 4
  commit_interval_commands: 3
quantization_point_batch_budget: 256
calibration:
  temperature: 0.7
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.snapshot_version == 4
    assert config.max_threads == 2
    assert config.thread_limit == 3
    assert config.distillation_temperature == 0.7
    assert config.layers["layer.a"].to_dict() == {
        "bits": 4,
        "group_size": 2,
        "fractional_bits": 3,
        "scheme": "asymmetric",
    }
    assert config.layers["layer.b"].bits == 8

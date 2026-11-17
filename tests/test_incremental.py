from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from softmax_layernorm.engine import Engine, compute_from_scratch
from softmax_layernorm.jsonutil import sanitize_json, sha256_json
from softmax_layernorm.models import CommandSpec


def _commands(items):
    return [
        CommandSpec(index=index, command_id=item.get("command_id", f"cmd-{index}"), raw=item)
        for index, item in enumerate(items)
    ]


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return np.asarray(left).dtype == np.asarray(right).dtype and np.asarray(left).tobytes() == np.asarray(right).tobytes()


def test_parameter_commands_invalidate_only_affected_operator(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=4, cols=5)
    commands = _commands(
        [
            {"type": "set_epsilon", "epsilon": 1e-3},
            {"type": "set_stabilize", "enabled": False},
            {"type": "set_softmax_axis", "axis": 0},
            {"type": "set_layernorm_axis", "axis": 0},
        ]
    )
    engine = Engine(
        loaded,
        commands,
        sha256_json([item.raw for item in commands]),
        output_dir=tmp_path / "parameter-output",
        fresh=True,
    )
    for command in commands:
        response = engine.handle_command(command)
        assert response["status"] == "ok"
    history = engine.reuse_recompute_history

    epsilon_stages = history[0]["calls"]["call-1"]
    assert epsilon_stages["softmax"]["recomputed_lanes"] == []
    assert epsilon_stages["layernorm"]["recomputed_lanes"] == [0, 1, 2, 3]

    stabilize_stages = history[1]["calls"]["call-1"]
    assert stabilize_stages["softmax"]["recomputed_lanes"] == [0, 1, 2, 3]
    assert stabilize_stages["layernorm"]["recomputed_lanes"] == []

    softmax_axis_stages = history[2]["calls"]["call-1"]
    assert softmax_axis_stages["softmax"]["recomputed_lanes"] == [0, 1, 2, 3, 4]
    assert softmax_axis_stages["layernorm"]["recomputed_lanes"] == []

    layernorm_axis_stages = history[3]["calls"]["call-1"]
    assert layernorm_axis_stages["layernorm"]["recomputed_lanes"] == [0, 1, 2, 3, 4]
    assert layernorm_axis_stages["softmax"]["recomputed_lanes"] == []
    engine.finalize()


def test_row_replacement_recomputes_only_replaced_rows(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=6, cols=4)
    commands = _commands(
        [
            {
                "type": "replace_rows",
                "start": 1,
                "end": 3,
                "values": [[1, 2, 3, 4], [5, 6, 7, 8]],
            }
        ]
    )
    engine = Engine(
        loaded,
        commands,
        sha256_json([item.raw for item in commands]),
        output_dir=tmp_path / "replace-output",
        fresh=True,
    )
    engine.handle_command(commands[0])
    stages = engine.reuse_recompute_history[0]["calls"]["call-1"]
    assert stages["softmax"]["recomputed_lanes"] == [1, 2]
    assert stages["layernorm"]["recomputed_lanes"] == [1, 2]
    assert stages["softmax"]["reused_lanes"] == [0, 3, 4, 5]
    assert stages["layernorm"]["reused_lanes"] == [0, 3, 4, 5]
    assert np.array_equal(engine.tensor[0], loaded.tensor[0])
    assert np.array_equal(engine.tensor[3], loaded.tensor[3])
    engine.finalize()


def test_interactive_edits_match_full_recompute_bit_for_bit(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(
        rows=6,
        cols=5,
        calls=[
            {
                "call_id": "a",
                "sequence": 1,
                "rows": 6,
                "cols": 5,
                "softmax_axis": 1,
                "layernorm_axis": 1,
                "epsilon": 1e-5,
                "stabilize": True,
                "reference_precision": "float64",
            },
            {
                "call_id": "b",
                "sequence": 2,
                "rows": 6,
                "cols": 5,
                "softmax_axis": 1,
                "layernorm_axis": 0,
                "epsilon": 1e-4,
                "stabilize": True,
            },
        ],
        tensor_spec={
            "shape": [6, 5],
            "seed": 7,
            "distribution": {"kind": "normal", "loc": 0, "scale": 2},
            "special_values": [{"row": 0, "col": 4, "value": 120.0}],
        },
    )
    commands = _commands(
        [
            {"type": "set_epsilon", "epsilon": 1e-6},
            {"type": "replace_rows", "start": 1, "end": 3, "values": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]},
            {"type": "set_stabilize", "call_id": "a", "enabled": False},
            {"type": "set_softmax_axis", "call_id": "a", "axis": 0},
            {"type": "replace_rows", "start": 4, "end": 6, "values": [[0, 1, 2, 3, 1000], [5, 4, 3, 2, 1]]},
            {"type": "set_layernorm_axis", "call_id": "a", "axis": 0},
            {"type": "set_epsilon", "call_id": "b", "epsilon": 2e-5},
        ]
    )
    engine = Engine(
        loaded,
        commands,
        sha256_json([item.raw for item in commands]),
        output_dir=tmp_path / "equivalence-output",
        fresh=True,
    )
    for command in commands:
        assert engine.handle_command(command)["status"] == "ok"
    report = engine.finalize()
    final_calls, final_tensor = engine.export_final()
    full = compute_from_scratch(
        final_calls,
        final_tensor,
        loaded.config,
        affected_rows=report["summary"]["affected_rows"],
    )
    assert report["summary"] == full.summary
    assert report["failures"] == sanitize_json([failure.to_dict() for failure in full.failures])
    incremental_by_id = {state.spec.call_id: state for state in engine.states}
    for state in full.states:
        incremental = incremental_by_id[state.spec.call_id]
        assert _arrays_equal(incremental.softmax_result, state.softmax_result)
        assert _arrays_equal(incremental.layernorm_result, state.layernorm_result)
        assert _arrays_equal(incremental.layernorm_mean, state.layernorm_mean)
        assert _arrays_equal(incremental.layernorm_variance, state.layernorm_variance)
        assert _arrays_equal(
            incremental.layernorm_epsilon_applied, state.layernorm_epsilon_applied
        )


def test_thread_count_does_not_change_results(make_run) -> None:
    loaded, _, _, _ = make_run(
        rows=20,
        cols=11,
        calls=[
            {
                "call_id": "a",
                "sequence": 1,
                "rows": 20,
                "cols": 11,
                "softmax_axis": 1,
                "layernorm_axis": 0,
                "epsilon": 1e-5,
                "stabilize": True,
            }
        ],
    )
    serial_config = loaded.config.__class__(**{**loaded.config.to_dict(), "thread_pool_workers": 1})
    parallel_config = loaded.config.__class__(**{**loaded.config.to_dict(), "thread_pool_workers": 8})
    serial = compute_from_scratch(loaded.calls, loaded.tensor, serial_config)
    parallel = compute_from_scratch(loaded.calls, loaded.tensor, parallel_config)
    assert _arrays_equal(serial.states[0].softmax_result, parallel.states[0].softmax_result)
    assert _arrays_equal(serial.states[0].layernorm_result, parallel.states[0].layernorm_result)
    assert _arrays_equal(serial.states[0].layernorm_mean, parallel.states[0].layernorm_mean)
    assert _arrays_equal(serial.states[0].layernorm_variance, parallel.states[0].layernorm_variance)
    assert [failure.to_dict() for failure in serial.failures] == [
        failure.to_dict() for failure in parallel.failures
    ]


def test_failure_order_is_lane_ascending_across_thread_counts(make_run) -> None:
    loaded, _, _, _ = make_run(
        rows=4,
        cols=4,
        calls=[
            {
                "call_id": "overflow-order",
                "sequence": 1,
                "rows": 4,
                "cols": 4,
                "softmax_axis": 0,
                "layernorm_axis": 1,
                "epsilon": 1e-5,
                "stabilize": False,
            }
        ],
        tensor_spec={
            "shape": [4, 4],
            "seed": 18,
            "distribution": {"kind": "normal", "loc": 0, "scale": 1},
            "special_values": [
                {"row": 0, "col": 3, "value": 1000.0},
                {"row": 1, "col": 1, "value": 1000.0},
            ],
        },
    )
    serial_config = loaded.config.__class__(**{**loaded.config.to_dict(), "thread_pool_workers": 1})
    parallel_config = loaded.config.__class__(**{**loaded.config.to_dict(), "thread_pool_workers": 8})
    serial = compute_from_scratch(loaded.calls, loaded.tensor, serial_config)
    parallel = compute_from_scratch(loaded.calls, loaded.tensor, parallel_config)
    assert [failure.lane for failure in serial.failures] == [1, 3]
    assert [failure.lane for failure in parallel.failures] == [1, 3]
    assert [failure.to_dict() for failure in serial.failures] == [
        failure.to_dict() for failure in parallel.failures
    ]


def test_invalid_runtime_command_records_business_codes(make_run, tmp_path: Path) -> None:
    loaded, _, _, _ = make_run(rows=3, cols=3)
    commands = _commands(
        [
            {"type": "set_epsilon", "epsilon": 0.0},
            {"type": "set_softmax_axis", "axis": 2},
            {"type": "not_a_command"},
        ]
    )
    engine = Engine(
        loaded,
        commands,
        sha256_json([item.raw for item in commands]),
        output_dir=tmp_path / "invalid-command-output",
        fresh=True,
    )
    codes = []
    for command in commands:
        response = engine.handle_command(command)
        assert response["status"] == "error"
        codes.append(response["business_codes"][0])
    assert codes == ["E_EPS_INVALID", "E_AXIS_OUT_OF_RANGE", "E_CONFIG_INVALID"]
    report = engine.finalize()
    assert [failure["code"] for failure in report["command_errors"]] == codes
    assert report["business_code_counts"]["E_EPS_INVALID"] == 1
    assert report["business_code_counts"]["E_AXIS_OUT_OF_RANGE"] == 1
    assert report["business_code_counts"]["E_CONFIG_INVALID"] == 1

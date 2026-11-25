from __future__ import annotations

from pathlib import Path

import numpy as np

from fixedtex.renderer import full_render_serial

from .helpers import (
    SAMPLE_CONFIG_LOW,
    SAMPLE_TEXTURE_V2,
    assert_session_matches_full,
    open_session,
    write_config,
)


def test_concurrency_does_not_change_artifacts_or_ledger(tmp_path: Path) -> None:
    config_one = write_config(tmp_path, thread_pool_size=1, commit_interval_commands=99, name="serial")
    config_four = write_config(tmp_path, thread_pool_size=4, commit_interval_commands=99, name="concurrent")
    serial = open_session(tmp_path, config_path=config_one, name="state_one")
    concurrent = open_session(tmp_path, config_path=config_four, name="state_four")
    command = {"command": "set_layer_param", "layer": 2, "frequency": 5.25, "amplitude": 0.75}
    serial_result = serial.apply_command(command)
    concurrent_result = concurrent.apply_command(command)
    assert serial_result["preview"]["max_active_tasks"] == 1
    assert concurrent_result["preview"]["max_active_tasks"] <= 4
    assert serial.state.composite is not None and concurrent.state.composite is not None
    assert np.array_equal(serial.state.composite.array, concurrent.state.composite.array)
    for index in serial.state.layers:
        assert serial.state.layers[index].key == concurrent.state.layers[index].key
        assert np.array_equal(serial.state.layers[index].quantized, concurrent.state.layers[index].quantized)
        assert np.array_equal(serial.state.layers[index].residual, concurrent.state.layers[index].residual)
    for name in serial.state.channels:
        assert serial.state.channels[name].key == concurrent.state.channels[name].key
        assert np.array_equal(serial.state.channels[name].array, concurrent.state.channels[name].array)
    assert serial.state.ledger == concurrent.state.ledger
    assert serial.state.failures == concurrent.state.failures
    assert serial.state.pending == concurrent.state.pending


def test_concurrency_does_not_change_failure_or_injection_order(tmp_path: Path) -> None:
    config_one = write_config(tmp_path, thread_pool_size=1, commit_interval_commands=99, name="fault_serial")
    config_four = write_config(tmp_path, thread_pool_size=4, commit_interval_commands=99, name="fault_concurrent")
    injections = [
        {"layer": 2, "stage": "layer_quantize", "code": "E_INJECTED_TWO"},
        {"layer": 0, "stage": "noise_synthesis", "code": "E_INJECTED_ZERO"},
        {"layer": 1, "stage": "noise_synthesis", "code": "E_INJECTED_ONE"},
    ]
    serial = open_session(tmp_path, config_path=config_one, name="fault_one", injections=injections)
    concurrent = open_session(
        tmp_path, config_path=config_four, name="fault_four", injections=injections
    )
    serial.advance(wait=True)
    concurrent.advance(wait=True)
    assert serial.state.failures == concurrent.state.failures
    assert serial.state.injection_hits == concurrent.state.injection_hits
    assert serial.state.ledger == concurrent.state.ledger
    assert serial.state.pending == concurrent.state.pending
    assert [item["layer"] for item in serial.state.failures if item["product"] == "layer"] == [0, 1, 2]
    assert [item["code"] for item in serial.state.injection_hits] == [
        "E_INJECTED_TWO",
        "E_INJECTED_ZERO",
        "E_INJECTED_ONE",
    ]


def test_budget_pending_resume_matches_full_render(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        source=SAMPLE_CONFIG_LOW,
        name="low-budget",
    )
    session = open_session(tmp_path, config_path=config_path, name="low_state")
    first = session.advance(wait=True)
    assert first["complete"] is False
    assert any(item["code"] == "E_POINT_BUDGET_PENDING" for item in session.state.pending)
    assert session.state.partials
    assert session.state.composite is None

    final = session.complete()
    assert final["complete"] is True
    assert not session.state.pending
    full_layers, full_channels, full_composite = full_render_serial(
        session.config, session.material, session.texture
    )
    assert_session_matches_full(session, full_layers, full_channels, full_composite)


def test_channel_weight_edit_reuses_layers_and_unaffected_channels(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, commit_interval_commands=99)
    session = open_session(tmp_path, config_path=config_path)
    session.apply_command(
        {"command": "set_channel_weight", "channel": "diffuse", "layer": 0, "weight": 0.5}
    )
    first_layer_keys = {index: item.key for index, item in session.state.layers.items()}
    first_channel_keys = {name: item.key for name, item in session.state.channels.items()}
    session.apply_command(
        {"command": "set_channel_weight", "channel": "roughness", "layer": 1, "weight": 0.7}
    )
    second_rows = [row for row in session.state.ledger if row.get("sequence") == 2]
    layer_rows = [row for row in second_rows if row.get("product") == "layer"]
    assert all(row["action"] == "reuse_hit" for row in layer_rows)
    assert first_layer_keys == {index: item.key for index, item in session.state.layers.items()}
    assert session.state.channels["diffuse"].key == first_channel_keys["diffuse"]
    assert session.state.channels["metallic"].key == first_channel_keys["metallic"]
    assert session.state.channels["normal"].key == first_channel_keys["normal"]
    assert session.state.channels["roughness"].key != first_channel_keys["roughness"]
    diffuse_row = next(row for row in second_rows if row.get("product") == "channel" and row.get("channel") == "diffuse")
    roughness_row = next(row for row in second_rows if row.get("product") == "channel" and row.get("channel") == "roughness")
    final_row = next(row for row in second_rows if row.get("product") == "composite")
    assert diffuse_row["action"] == "reuse_hit"
    assert roughness_row["action"] == "recompute"
    assert roughness_row["reason"] == "channel_weight_changed"
    assert roughness_row["references"]["layer_indices"] == [1, 2, 4]
    assert final_row["reason"] == "upstream_channel_changed"


def test_frequency_edit_invalidates_only_dependent_products(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, commit_interval_commands=99)
    session = open_session(tmp_path, config_path=config_path)
    session.apply_command({"command": "render"})
    first_layer_keys = {index: item.key for index, item in session.state.layers.items()}
    first_channel_keys = {name: item.key for name, item in session.state.channels.items()}
    session.apply_command({"command": "set_layer_param", "layer": 1, "frequency": 9.0})
    second_rows = [row for row in session.state.ledger if row.get("sequence") == 2]
    layer_rows = {row["index"]: row for row in second_rows if row.get("product") == "layer"}
    assert layer_rows[0]["action"] == "reuse_hit"
    assert layer_rows[1]["action"] == "recompute_complete"
    assert layer_rows[1]["reason"] == "layer_parameter_changed"
    assert layer_rows[2]["action"] == "reuse_hit"
    assert session.state.layers[0].key == first_layer_keys[0]
    assert session.state.layers[1].key != first_layer_keys[1]
    assert session.state.channels["metallic"].key == first_channel_keys["metallic"]
    assert session.state.channels["diffuse"].key != first_channel_keys["diffuse"]
    assert session.state.channels["roughness"].key != first_channel_keys["roughness"]
    assert session.state.channels["normal"].key != first_channel_keys["normal"]


def test_texture_replacement_invalidates_all_downstream_products(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, commit_interval_commands=99)
    session = open_session(tmp_path, config_path=config_path)
    session.apply_command({"command": "render"})
    first_layer_keys = {index: item.key for index, item in session.state.layers.items()}
    session.apply_command({"command": "replace_texture", "path": str(SAMPLE_TEXTURE_V2)})
    second_rows = [row for row in session.state.ledger if row.get("sequence") == 2]
    layer_rows = [row for row in second_rows if row.get("product") == "layer"]
    assert all(row["reason"] == "base_texture_changed" for row in layer_rows)
    assert all(session.state.layers[index].key != first_layer_keys[index] for index in session.state.layers)
    assert all(row["reason"] == "upstream_layer_changed" for row in second_rows if row.get("product") == "channel")


def test_quant_step_switch_changes_layer_integers_and_residuals(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, commit_interval_commands=99)
    session = open_session(tmp_path, config_path=config_path)
    session.apply_command({"command": "render"})
    raw_before = {index: item.raw.copy() for index, item in session.state.layers.items()}
    quant_before = {index: item.quantized.copy() for index, item in session.state.layers.items()}
    residual_before = {index: item.residual.copy() for index, item in session.state.layers.items()}
    session.apply_command({"command": "set_quant_step", "step": 8})
    assert session.state.composite is not None
    for index in session.state.layers:
        artifact = session.state.layers[index]
        assert np.array_equal(artifact.raw, raw_before[index])
        assert not np.array_equal(artifact.quantized, quant_before[index])
        assert not np.array_equal(artifact.residual, residual_before[index])
        assert artifact.quant_step == 8

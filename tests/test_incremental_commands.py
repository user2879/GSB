from __future__ import annotations

import json

from layerpack.engine import LayerPackEngine
from layerpack.interaction import CompressionSession

from conftest import EXAMPLES, ROOT, clone, write_json


def _final_payload(frame) -> dict:
    assert frame.payload["frame_type"] == "final"
    assert frame.payload["result"]["status"] in {"complete", "failed"}
    return frame.payload


def test_unchanged_rerun_reuses_everything_and_keeps_snapshot_version(
    sample_paths, tmp_path
):
    state_dir = tmp_path / "state"
    first = LayerPackEngine(
        sample_paths["weights"],
        sample_paths["calibration"],
        sample_paths["config"],
        state_dir,
    ).run()
    second = LayerPackEngine(
        sample_paths["weights"],
        sample_paths["calibration"],
        sample_paths["config"],
        state_dir,
    ).run()
    assert first.status == second.status == "complete"
    assert second.snapshot_version == first.snapshot_version
    assert second.diagnostics["reused"]["quant_parameters"] == sorted(
        second.diagnostics["completed_layers"]
    )
    assert second.diagnostics["reused"]["calibration_deviations"] == sorted(
        second.diagnostics["completed_layers"]
    )
    assert second.diagnostics["reused"]["deployment_manifest"] is True
    assert second.diagnostics["invalidated"]["quant_parameters"] == []
    assert second.diagnostics["invalidated"]["calibration_deviations"] == []


def test_layer_edit_invalidates_only_layer_and_downstream_calibration(
    sample_paths, tmp_path
):
    session = CompressionSession(
        sample_paths["weights"],
        sample_paths["calibration"],
        sample_paths["config"],
        tmp_path / "interactive",
    )
    initial = session.initialize()
    assert initial.status == "complete"
    frame = session.submit_command(
        {"type": "set_bits", "layer": "encoder.attn.k", "bits": 8},
        latency_budget_ms=10_000,
    )
    payload = _final_payload(frame)
    diagnostics = payload["result"]["diagnostics"]
    assert diagnostics["invalidated"]["quant_parameters"] == ["encoder.attn.k"]
    assert diagnostics["invalidated"]["dequantized_weights"] == ["encoder.attn.k"]
    assert "embed" in diagnostics["reused"]["quant_parameters"]
    assert diagnostics["invalidated"]["calibration_deviations"] == [
        "encoder.attn.k",
        "encoder.attn.q",
        "encoder.attn.v",
        "encoder.norm",
        "head",
    ]
    assert diagnostics["reused"]["quant_parameters"] == [
        "embed",
        "encoder.attn.q",
        "encoder.attn.v",
        "encoder.norm",
        "head",
    ]
    assert diagnostics["invalidated"]["deployment_manifest"] is True


def test_targeted_calibration_edit_invalidates_affected_and_downstream_only(
    sample_paths, base_calibration, tmp_path
):
    initial_calibration = clone(base_calibration)
    initial_calibration["samples"][0]["layers"] = ["encoder.attn.k"]
    calibration_path = write_json(
        tmp_path / "targeted-calibration.json", initial_calibration
    )
    session = CompressionSession(
        sample_paths["weights"],
        calibration_path,
        sample_paths["config"],
        tmp_path / "calibration-edit",
    )
    session.initialize()
    sample = dict(initial_calibration["samples"][0])
    sample["layers"] = ["encoder.attn.k"]
    sample["input"] = [0.01 * (index + 1) for index in range(8)]
    frame = session.submit_command(
        {"type": "replace_calibration_sample", "sample": sample},
        latency_budget_ms=10_000,
    )
    diagnostics = _final_payload(frame)["result"]["diagnostics"]
    assert diagnostics["reused"]["quant_parameters"] == sorted(
        diagnostics["completed_layers"]
    )
    assert diagnostics["invalidated"]["calibration_deviations"] == [
        "encoder.attn.k",
        "encoder.attn.q",
        "encoder.attn.v",
        "encoder.norm",
        "head",
    ]
    assert "embed" in diagnostics["reused"]["calibration_deviations"]


def test_eight_command_sequence_matches_blank_full_rebuild(sample_paths, tmp_path):
    commands = [
        json.loads(line)
        for line in (EXAMPLES / "commands-all-eight.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    session = CompressionSession(
        sample_paths["weights"],
        sample_paths["calibration"],
        sample_paths["config"],
        tmp_path / "incremental",
    )
    session.initialize()
    frames, final = session.apply_commands(commands, latency_budget_ms=10_000)
    assert all(frame.payload["frame_type"] == "final" for frame in frames)
    incremental = _final_payload(final)
    assert incremental["result"]["status"] == "complete"
    assert set(command["type"] for command in commands) == {
        "set_bits",
        "set_scheme",
        "set_group_size",
        "set_per_channel",
        "set_temperature",
        "set_soft_label_weight",
        "replace_layer_weights",
        "replace_calibration_sample",
    }

    exported = session.export_final_state(tmp_path / "final-inputs")
    full = LayerPackEngine(
        exported["weights"],
        exported["calibration"],
        exported["config"],
        tmp_path / "full",
        use_cache=False,
    ).run(full_rebuild=True)
    assert full.status == "complete"
    assert incremental["result"]["result"] == full.comparable()
    last_diagnostics = incremental["result"]["diagnostics"]
    # The last command only replaced a calibration sample. Every quantized
    # layer therefore had to be reused, while the manifest had to be rebuilt.
    assert last_diagnostics["reused"]["quant_parameters"] == sorted(
        last_diagnostics["completed_layers"]
    )
    assert last_diagnostics["invalidated"]["deployment_manifest"] is True

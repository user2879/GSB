from __future__ import annotations

import copy
import time

import pytest

from layerpack.engine import LayerPackEngine
from layerpack.errors import E_POINT_BUDGET_PENDING
from layerpack.interaction import CompressionSession
from layerpack.session import SessionStore

from conftest import clone, write_config


def test_snapshot_phases_and_restore_point(sample_paths, tmp_path):
    summary = LayerPackEngine(
        sample_paths["weights"],
        sample_paths["calibration"],
        sample_paths["config"],
        tmp_path / "state",
    ).run()
    assert summary.status == "complete"
    assert summary.restore_point is not None
    assert set(summary.restore_point["phases"]) == {
        "loading",
        "quantization",
        "calibration",
        "deployment",
    }
    assert summary.restore_point["completed_layers"] == sorted(
        summary.diagnostics["completed_layers"]
    )


def test_point_budget_pending_then_resumes_with_larger_budget(
    sample_paths, base_config, tmp_path
):
    low = clone(base_config)
    low["runtime"]["point_budget_per_batch"] = 8
    low_path = write_config(tmp_path / "low.yaml", low)
    state_dir = tmp_path / "state"
    pending = LayerPackEngine(
        sample_paths["weights"], sample_paths["calibration"], low_path, state_dir
    ).run()
    assert pending.status == "pending"
    assert E_POINT_BUDGET_PENDING in pending.business_codes
    assert pending.diagnostics["pending_queue"]
    assert pending.core["deployment_manifest"] is None

    high = clone(base_config)
    high_path = write_config(tmp_path / "high.yaml", high)
    resumed = LayerPackEngine(
        sample_paths["weights"], sample_paths["calibration"], high_path, state_dir
    ).run()
    assert resumed.status == "complete"
    assert resumed.core["deployment_manifest"] is not None


def test_interaction_budget_returns_interim_without_persisting_command(
    sample_paths, base_config, tmp_path
):
    config = clone(base_config)
    config["runtime"]["stage_delay_ms"] = 200
    config["runtime"]["interaction_latency_budget_ms"] = 5
    config_path = write_config(tmp_path / "delay.yaml", config)
    state_dir = tmp_path / "state"
    session = CompressionSession(
        sample_paths["weights"], sample_paths["calibration"], config_path, state_dir
    )
    started = time.monotonic()
    interim = session.submit_command(
        {"type": "set_bits", "layer": "head", "bits": 8}
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    assert interim.payload["frame_type"] == "interim"
    assert interim.payload["business_code"] == "E_INTERACT_BUDGET_PENDING"
    assert interim.payload["persisted"] is False
    assert elapsed_ms < 80.0
    assert interim.payload["affected_layers"] == ["head"]

    persisted_while_pending = SessionStore(state_dir).load()
    assert persisted_while_pending is not None
    assert persisted_while_pending.layers["head"].content_version == 6

    final = session.wait_for_final(interim)
    assert final.payload["frame_type"] == "final"
    assert final.payload["persisted"] is True
    assert final.payload["result"]["status"] == "complete"
    persisted_after = SessionStore(state_dir).load()
    assert persisted_after is not None
    assert persisted_after.layers["head"].content_version == 6
    assert (
        persisted_after.config["layers"]["head"]["bits"] == 8
    )


def test_latest_completed_summary_precedes_new_pending_command(
    sample_paths, base_config, tmp_path
):
    config = clone(base_config)
    config["runtime"]["stage_delay_ms"] = 150
    config_path = write_config(tmp_path / "delay.yaml", config)
    session = CompressionSession(
        sample_paths["weights"],
        sample_paths["calibration"],
        config_path,
        tmp_path / "state",
    )
    session.initialize()
    interim = session.submit_command(
        {"type": "set_soft_label_weight", "value": 0.1},
        latency_budget_ms=2,
    )
    assert interim.payload["frame_type"] == "interim"
    latest = interim.payload["latest_completed_summary"]
    assert latest is not None
    assert latest["completed_layer_count"] == 6
    assert interim.payload["progress"]["completed"] == 6
    final = session.wait_for_final(interim)
    assert final.payload["result"]["status"] == "complete"

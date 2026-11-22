from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from image_edit_workbench.config import load_config
from image_edit_workbench.errors import E_RECOMPUTE_BUDGET_PENDING
from image_edit_workbench.exchange import materialize_logical_state
from image_edit_workbench.pipeline import run_pipeline
from image_edit_workbench.runner import ScenarioRunner, build_snapshot_manifest, load_resume_state
from image_edit_workbench.snapshot import SnapshotStore
from image_edit_workbench.util import read_json, sha256_file

from .helpers import cold_render, read_rgb


def test_deterministic_render_and_metadata(workspace, tmp_path):
    config = workspace["config"]
    first = cold_render(
        list(workspace["regions"]),
        config=config,
        images=workspace["images"],
        work_root=tmp_path / "first",
    )[0]
    second = cold_render(
        list(workspace["regions"]),
        config=config,
        images=workspace["images"],
        work_root=tmp_path / "second",
    )[0]
    assert first.content_fingerprint == second.content_fingerprint
    assert first.stage_summaries == second.stage_summaries
    assert first.image_fingerprints == second.image_fingerprints
    for image_name in workspace["images"]:
        assert read_rgb(tmp_path / "first" / "outputs" / image_name) == read_rgb(
            tmp_path / "second" / "outputs" / image_name
        )
        assert sha256_file(tmp_path / "first" / "outputs" / image_name) == sha256_file(
            tmp_path / "second" / "outputs" / image_name
        )
    metadata = read_json(tmp_path / "first" / "outputs" / "metadata.json")
    assert metadata["style_quantization_levels"] == config.style_quantization_levels
    assert metadata["fixed_fraction_bits"] == config.fixed_fraction_bits
    assert metadata["rounding"] == "half_up"
    assert metadata["deterministic"] is True


def test_quantization_and_feather_change_real_pixels(workspace, tmp_path):
    base_config = workspace["config"]
    levels8 = base_config.updated(style_quantization_levels=8)
    levels16 = base_config.updated(style_quantization_levels=16)
    result8 = cold_render(
        list(workspace["regions"]), config=levels8, images=workspace["images"], work_root=tmp_path / "q8"
    )[0]
    result16 = cold_render(
        list(workspace["regions"]), config=levels16, images=workspace["images"], work_root=tmp_path / "q16"
    )[0]
    assert result8.image_fingerprints != result16.image_fingerprints
    assert result8.stage_summaries["style"] != result16.stage_summaries["style"]

    region_key = "scene_alpha.png#001"
    feather0 = [region.with_fields(feather_radius=region.feather_radius * 0) if region.region_key == region_key else region for region in workspace["regions"]]
    feather4 = [region.with_fields(feather_radius=region.feather_radius + 4) if region.region_key == region_key else region for region in workspace["regions"]]
    assert feather4[0].feather_radius <= base_config.feather_radius_max
    f0 = cold_render(feather0, config=base_config, images=workspace["images"], work_root=tmp_path / "f0")[0]
    f4 = cold_render(feather4, config=base_config, images=workspace["images"], work_root=tmp_path / "f4")[0]
    assert f0.stage_summaries["mask"][region_key] != f4.stage_summaries["mask"][region_key]
    assert read_rgb(tmp_path / "f0" / "outputs" / "scene_alpha.png") != read_rgb(
        tmp_path / "f4" / "outputs" / "scene_alpha.png"
    )


def test_concurrent_and_serial_results_match(workspace, tmp_path):
    config = workspace["config"]
    serial = cold_render(
        list(workspace["regions"]),
        config=config.updated(thread_pool_workers=1),
        images=workspace["images"],
        work_root=tmp_path / "serial",
    )[0]
    parallel = cold_render(
        list(workspace["regions"]),
        config=config.updated(thread_pool_workers=4),
        images=workspace["images"],
        work_root=tmp_path / "parallel",
    )[0]
    assert serial.image_fingerprints == parallel.image_fingerprints
    assert serial.stage_summaries == parallel.stage_summaries
    assert serial.content_fingerprint == parallel.content_fingerprint
    assert parallel.max_active_tasks <= 4


def test_single_region_and_stage_level_invalidation(workspace, tmp_path):
    commands = [
        {"id": "C1", "op": "noop"},
        {
            "id": "C2",
            "op": "update_region",
            "image": "scene_alpha.png",
            "region_index": 1,
            "changes": {"style_strength": "0.6125"},
        },
        {"id": "C3", "op": "set_config", "key": "style_quantization_levels", "value": 8},
        {"id": "C4", "op": "set_exchange_version", "version": 1},
    ]
    run = ScenarioRunner(
        sample_dir=workspace["sample_dir"],
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "outputs",
        reference_root=tmp_path / "reference",
    ).run(
        commands=commands,
        config=workspace["config"],
        exchange_version=2,
        drain_after=True,
        verify_each_command=False,
    )
    assert run.commands[1].result.recomputed_regions == ["scene_alpha.png#001"]
    assert len(run.commands[1].result.reused_regions) == 5
    alpha2_trace = next(trace for trace in run.commands[2].result.traces if trace.region_key == "scene_alpha.png#002")
    statuses = {stage.stage: stage.status for stage in alpha2_trace.stages}
    assert statuses["parse"] == "reused"
    assert statuses["mask"] == "reused"
    assert statuses["repair"] == "reused"
    assert statuses["attribute"] == "reused"
    assert statuses["style"] == "computed"
    assert statuses["compose"] == "computed"
    assert len(run.commands[3].result.recomputed_regions) == 6


def test_budget_pending_then_drain_and_snapshot_manifest(workspace, tmp_path):
    commands = [
        {"id": "B1", "op": "set_config", "key": "recompute_region_budget", "value": 25000},
        {"id": "B2", "op": "run"},
        {"id": "B3", "op": "run"},
        {"id": "B4", "op": "drain"},
    ]
    run = ScenarioRunner(
        sample_dir=workspace["sample_dir"],
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "outputs",
        reference_root=tmp_path / "reference",
    ).run(
        commands=commands,
        config=workspace["config"],
        exchange_version=2,
        drain_after=False,
        verify_each_command=False,
    )
    first = run.commands[0]
    assert first.result.pending_queue
    assert any(failure.code == E_RECOMPUTE_BUDGET_PENDING for failure in first.result.failures)
    for index, entry in enumerate(first.result.budget_log, start=1):
        assert entry["seq"] == index
    second = run.commands[1]
    assert second.result.budget_log[0]["priority"] == "pending"
    assert second.result.budget_log[0]["remaining_before"] == 25000
    final = run.commands[-1].result
    assert final.pending_queue == []
    assert sorted(final.region_records) == sorted(region.region_key for region in workspace["regions"])

    snapshot = run.snapshot_history[-1]
    manifest = read_json(Path(snapshot["path"]) / "manifest.json")
    assert manifest["global_version"] >= 1
    assert manifest["config_hash"]
    assert manifest["exchange_version"] == 2
    assert manifest["sidecar_hash"]
    assert manifest["completed_regions"] == sorted(final.region_records)
    assert manifest["artifact_index"]

    final_exchange = materialize_logical_state(
        run.final_state.regions,
        version=run.final_state.exchange_version,
        images=workspace["images"],
        manifest_keys={region.region_key for region in run.final_state.regions},
    )
    reference = cold_render(
        run.final_state.regions,
        config=run.final_state.config,
        images=workspace["images"],
        work_root=tmp_path / "reference-cold",
        version=run.final_state.exchange_version,
    )[0]
    assert final.image_fingerprints == reference.image_fingerprints
    assert final.stage_summaries == reference.stage_summaries
    assert final.content_fingerprint == reference.content_fingerprint


def test_command_sequence_incremental_equals_cold(workspace, tmp_path):
    commands = [
        {"id": "E1", "op": "noop"},
        {
            "id": "E2",
            "op": "update_region",
            "image": "scene_beta.png",
            "region_index": 1,
            "changes": {"rect": {"x": 34.5, "y": 59.25, "w": 60.0, "h": 70.0}},
        },
        {"id": "E3", "op": "set_config", "key": "style_quantization_levels", "value": 8},
        {"id": "E4", "op": "set_exchange_version", "version": 1},
    ]
    run = ScenarioRunner(
        sample_dir=workspace["sample_dir"],
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "outputs",
        reference_root=tmp_path / "reference",
    ).run(
        commands=commands,
        config=workspace["config"],
        exchange_version=2,
        drain_after=True,
        verify_each_command=True,
        keep_reference=False,
    )
    assert all(command.verification["semantic_equivalence"] for command in run.commands)
    assert all(command.verification["pixels_equal"] for command in run.commands)
    assert all(command.verification["stage_summaries_equal"] for command in run.commands)
    assert all(command.verification["content_fingerprint_equal"] for command in run.commands)


def test_resume_prioritizes_persisted_pending(workspace, tmp_path):
    first = ScenarioRunner(
        sample_dir=workspace["sample_dir"],
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "outputs",
        reference_root=tmp_path / "reference",
    ).run(
        commands=[
            {"id": "R1", "op": "set_config", "key": "recompute_region_budget", "value": 25000},
            {"id": "R2", "op": "run"},
        ],
        config=workspace["config"],
        exchange_version=2,
        drain_after=False,
        verify_each_command=False,
    )
    assert first.final_result.pending_queue
    resumed_state = load_resume_state(tmp_path / "state")
    assert resumed_state.config.recompute_region_budget == 25000
    resumed = ScenarioRunner(
        sample_dir=workspace["sample_dir"],
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "outputs",
        reference_root=tmp_path / "reference",
    ).run(
        commands=[{"id": "R3", "op": "run"}],
        config=workspace["config"],
        exchange_version=2,
        drain_after=False,
        verify_each_command=False,
        initial_state=resumed_state,
    )
    assert resumed.final_result.budget_log[0]["priority"] == "pending"
    assert resumed.resume["previous_pending"]

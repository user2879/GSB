from __future__ import annotations

from pathlib import Path

from gsscene import codes
from gsscene.compare import compare_cache_arrays, compare_reports
from gsscene.experiment import finalize_sample_from_edits
from gsscene.pipeline import ScenePipeline
from gsscene.sample import generate_sample
from gsscene.util import atomic_write_json, read_json
from gsscene.validation import materialize_injected_sample

from conftest import copy_sample, update_config, update_injections, write_edits


def test_incremental_equals_full_and_concurrent_equals_serial(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    incremental_pipeline = ScenePipeline(sample, tmp_path / "incremental", replay_edits=True)
    incremental_pipeline.run(0)
    for _ in range(8):
        incremental_pipeline.run(1)
    incremental = incremental_pipeline.run(0)

    final_sample = finalize_sample_from_edits(sample, tmp_path / "final_sample")
    full = ScenePipeline(final_sample, tmp_path / "full", replay_edits=False).run(0)
    serial = ScenePipeline(final_sample, tmp_path / "serial", replay_edits=False, worker_override=1).run(0)

    incremental_full = compare_reports(incremental, full)
    concurrent_serial = compare_reports(full, serial)
    assert incremental_full["equal"], incremental_full
    assert concurrent_serial["equal"], concurrent_serial
    assert compare_cache_arrays(tmp_path / "incremental", tmp_path / "full")["equal"]
    assert compare_cache_arrays(tmp_path / "full", tmp_path / "serial")["equal"]
    assert full["actual_invalidation"] == serial["actual_invalidation"]
    actions = incremental["actual_invalidation"]
    assert actions["recompute_kernels"] == []
    assert actions["recompute_tiles"] == []
    assert actions["recompute_composites"] == []
    assert actions["reused"] > 0
    assert full["budget_concurrency"]["configured"] == 4
    assert serial["budget_concurrency"]["effective"] == 1


def test_budget_ledger_and_pending_queue(tmp_path: Path) -> None:
    sample = generate_sample(tmp_path / "sample", width=16, height=16, views=2, force=True)
    config = read_json(sample / "config.json")
    config["kernel_budgets"] = [100, 2, 1]
    config["point_budget"] = 5
    atomic_write_json(sample / "config.json", config)
    report = ScenePipeline(sample, tmp_path / "work", replay_edits=False).run(0)
    failure_codes = {item["code"] for item in report["failures"]}
    assert codes.E_KERNEL_BUDGET_EXHAUSTED in failure_codes
    assert codes.E_POINT_BUDGET_PENDING in failure_codes
    entries = report["budget_ledger"]["kernel_scheduler"]["entries"]
    assert entries
    assert all(entry["remaining_kernel_budget"] >= 0 for entry in entries)
    assert all(entry["remaining_point_budget"] >= 0 for entry in entries)
    assert any(entry["status"] == codes.E_KERNEL_BUDGET_EXHAUSTED for entry in entries)
    assert any(entry["status"] == codes.E_POINT_BUDGET_PENDING for entry in entries)
    assert report["pending"]
    assert report["diagnostics"] == []


def test_injection_cases_are_isolated_and_all_hit(tmp_path: Path) -> None:
    sample = generate_sample(tmp_path / "sample", width=8, height=8, views=2, force=True)
    update_config(sample, kernel_budgets=[64, 64, 32])
    injections = [
        {"injection_id": "INJ-T", "kind": "truncate_image", "view_index": 0},
        {"injection_id": "INJ-C", "kind": "missing_camera", "view_index": 1},
        {"injection_id": "INJ-S", "kind": "zero_scale_decay", "level": 1},
        {"injection_id": "INJ-D", "kind": "density_over_budget", "view_index": 0, "level": 2, "params": {"density": 3.0}},
        {"injection_id": "INJ-Q", "kind": "quant_gear_out_of_range", "params": {"gear": "BOGUS"}},
    ]
    update_injections(sample, injections)
    expected = {
        "INJ-T": codes.E_IMAGE_TRUNCATED,
        "INJ-C": codes.E_CAMERA_MISSING,
        "INJ-S": codes.E_CONFIG_SCALE_DECAY,
        "INJ-D": codes.E_INJECTION_DENSITY_OVER_BUDGET,
        "INJ-Q": codes.E_CONFIG_QUANT_GEAR,
    }
    for injection in injections:
        injection_id = injection["injection_id"]
        injected_sample = materialize_injected_sample(
            sample,
            tmp_path / "injected" / injection_id,
            selected_injection_ids={injection_id},
        )
        report = ScenePipeline(injected_sample, tmp_path / "work" / injection_id, replay_edits=False).run(0)
        actual = {item["code"] for item in report["failures"] if item.get("injection_id") == injection_id}
        assert expected[injection_id] in actual, (injection_id, report["failures"])
        if expected[injection_id] in {codes.E_IMAGE_TRUNCATED, codes.E_CAMERA_MISSING}:
            assert report["diagnostics"]
            assert report["excluded_views"]


def test_snapshot_resume_and_force_rebuild(base_sample: Path, tmp_path: Path) -> None:
    sample = copy_sample(base_sample, tmp_path / "sample")
    write_edits(sample, [])
    pipeline = ScenePipeline(sample, tmp_path / "work", replay_edits=False)
    first = pipeline.run(0)
    manifest = first["snapshot_manifest"]
    assert manifest["version"] == 1
    assert manifest["config_hash"]
    assert manifest["camera_hash"]
    assert manifest["completed_views"] == [0, 1, 2, 3, 4, 5]
    assert manifest["completed_levels"] == [0, 1, 2]
    assert (tmp_path / "work" / "snapshots" / "committed" / "kernel" / "STAGE.json").exists()

    resumed = ScenePipeline(sample, tmp_path / "work", replay_edits=False).run(0)
    assert resumed["resume_mode"] == "RESUME"
    assert resumed["snapshot_manifest"]["version"] == 2
    assert resumed["actual_invalidation"]["recompute_kernels"] == []

    rebuilt = ScenePipeline(sample, tmp_path / "work", replay_edits=False, force_rebuild=True).run(0)
    assert rebuilt["resume_mode"] == "REBUILD"
    assert rebuilt["actual_invalidation"]["reuse_kernels"] == []
    assert rebuilt["actual_invalidation"]["recompute_kernels"]


def test_failure_list_order_is_stable(tmp_path: Path) -> None:
    sample = generate_sample(tmp_path / "sample", width=8, height=8, views=2, force=True)
    update_config(sample, scale_decay=0.0, quant_gear="BOGUS", levels=0)
    first = ScenePipeline(sample, tmp_path / "a", replay_edits=False).run(0)["failures"]
    second = ScenePipeline(sample, tmp_path / "b", replay_edits=False).run(0)["failures"]
    assert first == second
    keys = [(item["source"], item["view_index"], item["level"], item["code"]) for item in first]
    assert keys == sorted(keys, key=lambda item: (item[0], -1 if item[1] is None else item[1], -1 if item[2] is None else item[2], item[3]))

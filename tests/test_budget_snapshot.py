from __future__ import annotations

from pathlib import Path

from qcolumn_diag.cache import SnapshotStore
from qcolumn_diag.pipeline import run_workbench
from tests.helpers import base_config, base_sample, run_in_dir, write_json


def _five_row_sample(columns: int, *, missing: list[list[int]] | None = None) -> dict:
    sample = base_sample()
    tensor = sample["tensors"]["t0"]
    tensor["shape"] = [5, columns]
    tensor["column_offsets"] = [float(index) for index in range(columns)]
    tensor["column_scales"] = [1.0] * columns
    tensor["missing_values"] = missing or []
    tensor["injections"] = []
    return sample


def test_budget_pending_then_resume_and_single_thread_order(tmp_path: Path) -> None:
    config = base_config(
        shape=[5, 3],
        column_groups=[
            {"group_id": "a", "columns": [0]},
            {"group_id": "b", "columns": [1]},
            {"group_id": "c", "columns": [2]},
        ],
        calibration_element_budget=3,
        max_workers=2,
    )
    sample = _five_row_sample(3, missing=[[1, 2]])
    state = tmp_path / "state"
    config_path = write_json(tmp_path / "config.json", config)
    sample_path = write_json(tmp_path / "sample.json", sample)
    initial = run_workbench(
        config_path=config_path,
        sample_path=sample_path,
        state_dir=state,
        report_path=tmp_path / "budget_initial_report.json",
    )
    assert initial["run"]["complete"] is False
    assert initial["summary"]["column_count_completed"] == 0
    assert initial["cache"]["pending_columns"] == [
        "t0:c000000",
        "t0:c000001",
        "t0:c000002",
    ]
    pending_failures = [
        item
        for item in initial["failures"]
        if item["error_code"] == "E_CALIB_BUDGET_PENDING"
    ]
    assert [item["column"] for item in pending_failures] == [0, 1, 2]
    assert initial["run"]["max_workers_observed"] <= 2

    config["calibration_element_budget"] = 10
    write_json(config_path, config)
    resumed = run_workbench(
        config_path=config_path,
        sample_path=sample_path,
        state_dir=state,
        report_path=tmp_path / "budget_resumed_report.json",
    )
    assert resumed["run"]["complete"] is True
    assert [column["column_id"] for column in resumed["columns"]] == [
        "t0:c000000",
        "t0:c000001",
        "t0:c000002",
    ]
    assert resumed["cache"]["pending_columns"] == []
    snapshot, _ = SnapshotStore(state).load()
    assert snapshot["complete"] is True
    assert snapshot["completed_columns"] == [
        "t0:c000000",
        "t0:c000001",
        "t0:c000002",
    ]

    single_config = dict(config)
    single_config["max_workers"] = 1
    single_report, _, _ = run_in_dir(
        tmp_path, single_config, sample, name="single_thread"
    )
    assert single_report["columns"] == resumed["columns"]
    assert single_report["sensitivity"] == resumed["sensitivity"]
    assert single_report["failures"] == resumed["failures"]
    assert single_report["run"]["max_workers_observed"] <= 1


def test_snapshot_generations_and_version_match_resume(tmp_path: Path) -> None:
    config = base_config(
        shape=[5, 4],
        column_groups=[{"group_id": "all", "columns": [0, 1, 2, 3]}],
        calibration_element_budget=5,
        snapshot_commit_interval_batches=2,
        max_workers=2,
    )
    sample = _five_row_sample(4)
    state = tmp_path / "state"

    config_path = tmp_path / "config.json"
    sample_path = tmp_path / "sample.json"
    write_json(config_path, config)
    write_json(sample_path, sample)
    report = run_workbench(
        config_path=config_path,
        sample_path=sample_path,
        state_dir=state,
        report_path=tmp_path / "report.json",
    )
    assert report["run"]["batch_count"] == 4
    snapshot_dirs = sorted((state / "snapshots").glob("v*"))
    assert len(snapshot_dirs) >= 2
    assert all((path / "snapshot.json").is_file() for path in snapshot_dirs)
    current = (state / "CURRENT").read_text(encoding="utf-8").strip()
    assert current == snapshot_dirs[-1].name
    store = SnapshotStore(state)
    snapshot, generation = store.load()
    assert generation == int(current[1:])
    assert snapshot["run_version"] == report["run"]["run_version"]
    assert snapshot["columns"]
    assert snapshot["completed_columns"]

    rerun = run_workbench(
        config_path=config_path,
        sample_path=sample_path,
        state_dir=state,
        report_path=tmp_path / "report2.json",
    )
    assert rerun["run"]["resume_status"] == "version_match_reuse"
    assert rerun["cache"]["recomputed_columns"] == []
    assert len(rerun["cache"]["reused_columns"]) == 4

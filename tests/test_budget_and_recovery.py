from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

from imputation_workbench.engine import WorkbenchEngine, compare_semantic_states
from imputation_workbench.snapshot import SnapshotStore

from conftest import load_state, make_paths


def _cell_prefix(cells: dict):
    return {
        row_id: {column: dict(cell) for column, cell in column_map.items()}
        for row_id, column_map in cells.items()
    }


def test_budget_pending_resumes_same_queue_and_matches_one_shot(workspace) -> None:
    budget_paths = make_paths(workspace, "budgeted")
    first = WorkbenchEngine(budget_paths).run(fresh=True, budget_override=3, max_slices=1)
    assert first["status"] == "pending"
    assert first["report"]["imputation"]["pending_count"] > 0
    assert "E_PATCH_BUDGET_PENDING" in first["report"]["business_codes"]
    state1, _ = load_state(budget_paths)
    pending1 = state1["pending_jobs"]
    assert pending1 == sorted(pending1, key=lambda job: (job["column"], job["row_id"]))
    prefix1 = _cell_prefix(state1["imputations"])

    WorkbenchEngine(budget_paths).run(budget_override=3, max_slices=1)
    state2, _ = load_state(budget_paths)
    pending_after = {(job["column"], job["row_id"]) for job in state2["pending_jobs"]}
    assert pending_after == {
        (job["column"], job["row_id"]) for job in pending1[3:]
    }
    for row_id, column_map in prefix1.items():
        assert state2["imputations"][row_id] == column_map

    final_result = WorkbenchEngine(budget_paths).run(budget_override=3)
    assert final_result["status"] == "complete"
    budget_state, _ = load_state(budget_paths)

    one_shot_paths = make_paths(workspace, "one-shot")
    WorkbenchEngine(one_shot_paths).run(fresh=True, budget_override=10_000)
    one_shot_state, _ = load_state(one_shot_paths)
    assert compare_semantic_states(budget_state, one_shot_state)["equal"]


def test_hard_kill_after_complete_snapshot_recovers(workspace) -> None:
    killed_paths = make_paths(workspace, "killed")
    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "imputation_workbench",
        "run",
        "--workspace",
        str(workspace),
        "--state",
        str(killed_paths.state),
        "--fresh",
        "--simulate-hard-kill-after-commit",
        "1",
    ]
    killed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    assert killed.returncode == 137, (killed.returncode, killed.stdout, killed.stderr)
    manifest = SnapshotStore(killed_paths.state).load_manifest()
    assert manifest is not None
    assert manifest["global_version"] == 1
    assert manifest["in_progress_batch"] is not None

    resumed = WorkbenchEngine(killed_paths).run()
    assert resumed["status"] == "complete"
    resumed_state, _ = load_state(killed_paths)

    control_paths = make_paths(workspace, "control")
    WorkbenchEngine(control_paths).run(fresh=True)
    control_state, _ = load_state(control_paths)
    assert compare_semantic_states(resumed_state, control_state)["equal"]


def test_graceful_stop_commits_report_then_resumes(workspace) -> None:
    paths = make_paths(workspace, "graceful")
    stopped = WorkbenchEngine(paths).run(fresh=True, stop_requested=lambda: True)
    assert stopped["status"] == "interrupted"
    report_path = paths.state / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    manifest = SnapshotStore(paths.state).load_manifest()
    assert manifest is not None
    completed_report = WorkbenchEngine(paths).run()
    assert completed_report["status"] == "complete"


def test_snapshot_interval_controls_batch_commit_cadence(workspace, tmp_path) -> None:
    variant = tmp_path / "interval"
    shutil.copytree(workspace, variant)
    config_path = variant / "imputation.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["snapshot_interval_batches"] = 2
    config["batch_cell_budget"] = 10_000
    with config_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    paths = make_paths(variant, "work", injections=False)
    result = WorkbenchEngine(paths).run(fresh=True)
    assert result["status"] == "complete"
    manifest = SnapshotStore(paths.state).load_manifest()
    assert manifest is not None
    assert manifest["snapshot_interval_batches"] == 2
    assert manifest["completed_batch_count"] == 4
    generations = [
        path
        for path in (paths.state / "generations").iterdir()
        if path.is_dir() and path.name.startswith("v")
    ]
    assert len(generations) == 2


def test_budget_widening_in_config_resumes_without_full_rebuild(workspace, tmp_path) -> None:
    def make_variant(name: str, budget: int) -> Path:
        target = tmp_path / name
        shutil.copytree(workspace, target)
        config_path = target / "imputation.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["batch_cell_budget"] = budget
        with config_path.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        return target

    resumed_workspace = make_variant("resumed", 3)
    resumed_paths = make_paths(resumed_workspace, "work", injections=False)
    pending = WorkbenchEngine(resumed_paths).run(fresh=True, max_slices=1)
    assert pending["status"] == "pending"

    config_path = resumed_workspace / "imputation.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["batch_cell_budget"] = 10_000
    with config_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    completed = WorkbenchEngine(resumed_paths).run()
    assert completed["status"] == "complete"
    resumed_state, _ = load_state(resumed_paths)

    control_workspace = make_variant("control-budget", 10_000)
    control_paths = make_paths(control_workspace, "work", injections=False)
    WorkbenchEngine(control_paths).run(fresh=True)
    control_state, _ = load_state(control_paths)
    assert compare_semantic_states(resumed_state, control_state)["equal"]

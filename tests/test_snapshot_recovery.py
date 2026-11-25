from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

from gsb_index.codes import (
    E_RECOVERY_CONFIG_CHANGED,
    E_RECOVERY_ORPHAN_TEMP,
)
from gsb_index.config import load_project
from gsb_index.edits import project_to_config
from gsb_index.engine import run_build
from gsb_index.snapshot import SnapshotStore


def test_staged_snapshot_is_atomic_and_manifest_is_complete(sample_config, tmp_path):
    result = run_build(sample_config, tmp_path / "work", mode="full", workers=3, global_version=1)
    assert result.committed is True
    loaded = SnapshotStore(tmp_path / "work").load_current()
    assert loaded is not None
    manifest = loaded.manifest
    assert manifest["global_version"] == 1
    assert manifest["index_revision"] == load_project(sample_config)[0].current_revision
    assert manifest["completed_doc_cursor"] == manifest["document_count"]
    assert manifest["consumed_budget"] == loaded.ledger["entries_count"]
    assert manifest["config_hash"] == result.config_hash
    assert sorted(loaded.summaries) == ["shard-a", "shard-b", "shard-c"]
    assert loaded.recall == result.recall


def test_forced_stage_termination_leaves_orphan_and_recovery_reports_it(sample_config, tmp_path):
    workdir = tmp_path / "work"
    run_build(sample_config, workdir, mode="full", workers=3, global_version=1)
    staged = run_build(sample_config, workdir, mode="incremental", workers=3, global_version=2, commit=False)
    assert staged.committed is False
    assert staged.staging_path is not None
    assert SnapshotStore(workdir).current_snapshot_id() == "snap-00000001-" + staged.config_hash[:10]
    recovered = run_build(sample_config, workdir, mode="incremental", workers=3, global_version=2)
    assert recovered.committed is True
    orphan_codes = [issue["code"] for issue in recovered.recovery["issues"]]
    assert E_RECOVERY_ORPHAN_TEMP in orphan_codes
    assert recovered.recovery["orphan_directories"]


def test_configuration_hash_change_forces_rebuild_decision(sample_project, write_config, tmp_path):
    initial_config = write_config(project_to_config(sample_project), "initial.json")
    run_build(initial_config, tmp_path / "work", mode="full", workers=3, global_version=1)
    changed = copy.deepcopy(project_to_config(sample_project))
    changed["scoring"]["quantum_step"] = 8
    changed_config = write_config(changed, "changed.json")
    result = run_build(changed_config, tmp_path / "work", mode="incremental", workers=3, global_version=2)
    codes = [issue["code"] for issue in result.recovery["issues"]]
    assert E_RECOVERY_CONFIG_CHANGED in codes
    assert result.recovery["decision"] == "rebuild"


def test_renamed_snapshot_without_pointer_is_detected_as_orphan(sample_config, tmp_path):
    workdir = tmp_path / "work"
    run_build(sample_config, workdir, mode="full", workers=3, global_version=1)
    staged = run_build(sample_config, workdir, mode="incremental", workers=3, global_version=2, commit=False)
    store = SnapshotStore(workdir)
    os.replace(staged.staging_path, store.root / staged.snapshot_id)
    recovered = run_build(sample_config, workdir, mode="incremental", workers=3, global_version=2)
    assert staged.snapshot_id in recovered.recovery["orphan_directories"]
    assert E_RECOVERY_ORPHAN_TEMP in [issue["code"] for issue in recovered.recovery["issues"]]


def test_cli_simulated_termination_exit_then_recovers(sample_config, tmp_path):
    workdir = tmp_path / "cli-work"
    command = [
        sys.executable,
        "-m",
        "gsb_index.cli",
        "build",
        "--config",
        str(sample_config),
        "--workdir",
        str(workdir),
        "--mode",
        "incremental",
        "--workers",
        "3",
        "--global-version",
        "1",
        "--simulate-termination",
    ]
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(command, cwd=str(project_root), capture_output=True, text=True, timeout=30)
    assert completed.returncode == 9, completed.stderr
    store = SnapshotStore(workdir)
    assert store.orphan_directories()
    recovered = run_build(sample_config, workdir, mode="incremental", workers=3, global_version=1)
    assert recovered.committed is True
    assert E_RECOVERY_ORPHAN_TEMP in [issue["code"] for issue in recovered.recovery["issues"]]

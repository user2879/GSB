import os

import pytest

from lora_bench.errors import E_CONFIG_MISMATCH, WorkbenchError
from lora_bench.pipeline import run_pipeline
from lora_bench.snapshot import SnapshotStore
from lora_bench.utils import read_json


def test_manifest_fields_and_stage_layout(input_dir, workspace, base_config):
    outcome = run_pipeline(input_dir, workspace, base_config)
    manifest = outcome["manifest"]
    for key in (
        "version",
        "config_hash",
        "base_fingerprint",
        "completed_tasks",
        "stages",
    ):
        assert key in manifest
    current = os.path.join(workspace, "current")
    assert os.path.exists(os.path.join(current, "manifest.json"))
    assert os.path.exists(os.path.join(current, "forward", "index.json"))
    assert os.path.exists(os.path.join(current, "validate", "summary.json"))
    for task in manifest["completed_tasks"]:
        assert os.path.exists(os.path.join(current, "train", f"{task}.json"))
        assert os.path.exists(os.path.join(current, "train", f"{task}.npz"))


def test_versions_are_frozen_on_each_commit(input_dir, workspace, base_config):
    run_pipeline(input_dir, workspace, base_config)
    run_pipeline(input_dir, workspace, base_config)
    store = SnapshotStore(workspace)
    versions = store.list_versions()
    # commit_interval=2 over 4 tasks => 1 intermediate + 1 final per run.
    assert versions == ["v000001", "v000002", "v000003", "v000004"]
    for version in versions:
        assert os.path.exists(
            os.path.join(workspace, "versions", version, "manifest.json")
        )
    intermediate = read_json(
        os.path.join(workspace, "versions", "v000001", "manifest.json")
    )
    assert intermediate["final"] is False
    assert len(intermediate["completed_tasks"]) == base_config["commit_interval"]
    final = read_json(
        os.path.join(workspace, "versions", "v000002", "manifest.json")
    )
    assert final["final"] is True
    assert len(final["completed_tasks"]) == 4


def test_strict_resume_rejects_config_change(
    input_dir, workspace, base_config
):
    run_pipeline(input_dir, workspace, base_config)
    changed = dict(base_config, rank=4)
    with pytest.raises(WorkbenchError) as info:
        run_pipeline(
            input_dir, workspace, changed, strict_resume=True
        )
    assert info.value.code == E_CONFIG_MISMATCH


def test_non_strict_run_never_mixes_artifacts(
    input_dir, workspace, base_config
):
    first = run_pipeline(input_dir, workspace, base_config)
    changed = dict(base_config, learning_rate=0.05)
    first_final_version = first["manifest"]["version"]
    second = run_pipeline(input_dir, workspace, changed)
    assert second["manifest"]["config_hash"] != first["manifest"]["config_hash"]
    assert second["manifest"]["version"] > first_final_version
    # Every current artifact must carry the new config hash lineage.
    manifest = read_json(os.path.join(workspace, "current", "manifest.json"))
    assert manifest["config_hash"] == second["manifest"]["config_hash"]
    assert second["reconciliation"]["mismatch_fields"] == ["config_hash"]


def test_staging_directory_is_cleaned_after_commit(
    input_dir, workspace, base_config
):
    run_pipeline(input_dir, workspace, base_config)
    staging = os.path.join(workspace, "staging")
    leftovers = [
        name
        for name in os.listdir(staging)
        if os.path.isdir(os.path.join(staging, name))
    ]
    assert leftovers == []

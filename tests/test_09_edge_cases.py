import os

import pytest

from lora_bench.errors import E_CONFIG_MISMATCH, WorkbenchError
from lora_bench.pipeline import run_pipeline
from lora_bench.snapshot import SnapshotStore
from lora_bench.utils import read_json


def test_commit_interval_one_freezes_every_task(input_dir, workspace, base_config):
    config = dict(base_config, commit_interval=1)
    outcome = run_pipeline(input_dir, workspace, config)
    versions = SnapshotStore(workspace).list_versions()
    assert len(versions) == len(outcome["manifest"]["completed_tasks"])
    for index, version in enumerate(versions, start=1):
        manifest = read_json(
            os.path.join(workspace, "versions", version, "manifest.json")
        )
        assert len(manifest["completed_tasks"]) == index
        assert manifest["final"] is (index == len(versions))


def test_strict_mismatch_does_not_publish_new_version(
    input_dir, workspace, base_config
):
    run_pipeline(input_dir, workspace, base_config)
    before = SnapshotStore(workspace).list_versions()
    with pytest.raises(WorkbenchError) as info:
        run_pipeline(
            input_dir,
            workspace,
            dict(base_config, rank=4),
            strict_resume=True,
        )
    assert info.value.code == E_CONFIG_MISMATCH
    after = SnapshotStore(workspace).list_versions()
    assert before == after


def test_unknown_target_layer_rejected_before_training(
    input_dir, workspace, base_config
):
    from lora_bench.errors import E_TARGET_LAYER_UNKNOWN

    config = dict(base_config, target_layers=[7])
    with pytest.raises(WorkbenchError) as info:
        run_pipeline(input_dir, workspace, config)
    assert info.value.code == E_TARGET_LAYER_UNKNOWN


def test_concurrency_change_alone_does_not_invalidate(
    input_dir, workspace, base_config
):
    run_pipeline(input_dir, workspace, base_config)
    outcome = run_pipeline(
        input_dir, workspace, dict(base_config, concurrency=4)
    )
    assert outcome["invalidation"]["invalidated"] == []
    assert sorted(outcome["reused_tasks"]) == sorted(
        outcome["manifest"]["completed_tasks"]
    )

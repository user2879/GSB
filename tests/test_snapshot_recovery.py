import json

import pytest

from mlimgedit.config import load_config_dict
from mlimgedit.pipeline import Pipeline
from mlimgedit.runner import comparable_state, full_run, run_plan
from mlimgedit.snapshot import SnapshotStore
from mlimgedit.plan import EditPlan
from scripts.bootstrap_demo import build_config, build_edit_plan


def _prepare_plan(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(build_config(threads=2, budget=50000), sort_keys=True), encoding="utf-8"
    )
    (root / "edit_plan.json").write_text(
        json.dumps(build_edit_plan(), sort_keys=True), encoding="utf-8"
    )
    from mlimgedit.datagen import generate_dataset

    config = load_config_dict(json.loads((root / "config.json").read_text()))
    generate_dataset(root / "samples", config, 12)
    return EditPlan.load(root / "edit_plan.json")


def test_manifest_tracks_versions_and_cursor(tmp_path):
    plan = _prepare_plan(tmp_path)
    run_plan(plan, tmp_path / "work", tmp_path / "store")
    manifest = SnapshotStore(tmp_path / "store").load_manifest()
    assert manifest["global_version"] == len(plan.batches)
    assert manifest["batch_cursor"] == len(plan.batches)
    assert manifest["label_set_fingerprint"]
    for stage, state in manifest["stages"].items():
        assert state["version"]
        assert isinstance(state["keys"], dict)


def test_interrupted_run_resumes_without_corruption(tmp_path):
    plan = _prepare_plan(tmp_path)
    store_path = tmp_path / "store"
    store = SnapshotStore(store_path)
    calls = {"n": 0}

    def interruption(batch_index, stage):
        # Raise once on the third batch right before scoring commits, then
        # disarm so the resume loop can complete against the same pipeline.
        calls["n"] += 1
        if batch_index == 2 and stage == "scoring":
            pipeline.interruption = None
            raise KeyboardInterrupt("simulated SIGINT")

    pipeline = Pipeline(store, interruption=interruption)
    from mlimgedit.plan import Workspace
    from mlimgedit.config import load_config

    workspace = Workspace(
        tmp_path / "work", load_config(plan.base_config_path), plan.base_sample_dir
    )
    with pytest.raises(KeyboardInterrupt):
        for index, batch in enumerate(plan.batches):
            config = workspace.apply_batches(plan.batches, max_index=index)
            pipeline.run_batch(config, workspace.samples_dir, index)

    manifest = store.load_manifest()
    # Batches 0 and 1 committed; the interrupted batch left no partial manifest.
    assert manifest["batch_cursor"] == 2

    # Resume from the committed cursor and finish normally.
    resumed_reports = []
    for index in range(2, len(plan.batches)):
        config = workspace.apply_batches(plan.batches, max_index=index)
        resumed_reports.append(pipeline.run_batch(config, workspace.samples_dir, index))
    final_manifest = store.load_manifest()
    assert final_manifest["batch_cursor"] == len(plan.batches)

    full_config, full_samples = __import__(
        "mlimgedit.runner", fromlist=["rebuild_final_state"]
    ).rebuild_final_state(plan, tmp_path / "final")
    full_report = full_run(full_config, full_samples, tmp_path / "full_store")
    assert comparable_state(resumed_reports[-1]) == comparable_state(full_report)


def test_commit_compares_versions_instead_of_blind_overwrite(tmp_path):
    plan = _prepare_plan(tmp_path)
    run_plan(plan, tmp_path / "work", tmp_path / "store")
    store = SnapshotStore(tmp_path / "store")
    manifest_before = store.load_manifest()
    # Re-running the identical final state from the same store must not create
    # new artifact files because content-addressed ids are unchanged.
    artifact_dir = store.committed_artifacts
    before_files = {path.name for path in artifact_dir.glob("*.json")}
    pipeline = Pipeline(store)
    from mlimgedit.plan import Workspace
    from mlimgedit.config import load_config

    workspace = Workspace(
        tmp_path / "work2", load_config(plan.base_config_path), plan.base_sample_dir
    )
    final_config = workspace.apply_batches(plan.batches)
    pipeline.run_batch(final_config, workspace.samples_dir, len(plan.batches))
    after_files = {path.name for path in artifact_dir.glob("*.json")}
    assert before_files == after_files
    assert manifest_before["stages"].keys() == store.load_manifest()["stages"].keys()


def test_snapshot_interval_gates_commit_cadence():
    from mlimgedit.config import load_config_dict

    config = load_config_dict(build_config(snapshot_every=2))
    pipeline = Pipeline(SnapshotStore.__new__(SnapshotStore))
    flags = [pipeline.should_commit(config, index, total_batches=5) for index in range(5)]
    assert flags == [False, True, False, True, True]

    config_one = load_config_dict(build_config(snapshot_every=1))
    pipeline_one = Pipeline(SnapshotStore.__new__(SnapshotStore))
    assert all(pipeline_one.should_commit(config_one, i, 3) for i in range(3))

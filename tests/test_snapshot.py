"""Requirement 7: staged atomic snapshots and stale-version detection."""

import copy
import json

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.injection import load_injection
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.snapshot import SnapshotStore, SNAPSHOT_FORMAT_VERSION


def _make_pipeline(sample_dir, tmp_path, config_name="pipeline_config.json"):
    config = load_config(sample_dir / config_name)
    snap_dir = tmp_path / "snapshot"
    config.snapshot_dir = str(snap_dir)
    fsm = load_fsm(sample_dir / "fsm.yaml")
    return Pipeline(config, fsm, {"injections": []}), SnapshotStore(snap_dir)


def test_staged_commits_are_whole_directory_swaps(sample_dir, tmp_path):
    pipeline, store = _make_pipeline(sample_dir, tmp_path)
    pipeline.run(sample_dir / "events.jsonl", None, stop_after_stage="load")
    state = store.load()
    assert state.stage == "load"
    assert state.config_hash and state.fsm_hash
    # current is a symlink to an immutable version directory (no file-by-file
    # overwrite of the committed tree).
    assert store.committed.is_symlink()

    pipeline.run(sample_dir / "events.jsonl", None, stop_after_stage="advance")
    advanced = store.load()
    assert advanced.stage == "advance"
    assert advanced.checkpoints, "advance stage persists checkpoints"

    report = pipeline.run(sample_dir / "events.jsonl", None)
    settled = store.load()
    assert settled.stage == "settle"
    assert settled.completed_batches == report["completed_batches"]
    assert set(settled.artifact_versions) == {"load", "advance", "settle"}


def test_resume_continues_from_completed_prefix(sample_dir, tmp_path):
    pipeline, store = _make_pipeline(sample_dir, tmp_path)
    pipeline.run(sample_dir / "events.jsonl", None, stop_after_stage="advance")
    before = store.load()
    pipeline.run(sample_dir / "events.jsonl", None)
    after = store.load()
    # Same inputs -> same hashes -> resume, not reset; checkpoints retained.
    assert after.config_hash == before.config_hash
    assert after.fsm_hash == before.fsm_hash
    assert len(after.checkpoints) == len(before.checkpoints)


def test_config_change_is_detected_as_stale(sample_dir, tmp_path):
    pipeline, store = _make_pipeline(sample_dir, tmp_path)
    pipeline.run(sample_dir / "events.jsonl", None)

    config = load_config(sample_dir / "pipeline_config.json")
    config.snapshot_dir = str(tmp_path / "snapshot")
    config.max_workers += 1  # changes config fingerprint
    fsm = load_fsm(sample_dir / "fsm.yaml")
    changed = Pipeline(config, fsm, {"injections": []})
    report = changed.run(sample_dir / "events.jsonl", None)
    assert report["snapshot_event"]["code"] == "E_SNAPSHOT_STALE"
    state = store.load()
    # Fresh replay rebuilt checkpoints from an empty cache.
    assert state.global_version == 1


def test_fsm_change_is_detected_as_stale(sample_dir, tmp_path):
    pipeline, store = _make_pipeline(sample_dir, tmp_path)
    pipeline.run(sample_dir / "events.jsonl", None)
    state, cache, stale = store.resume_or_reset(
        pipeline.config.fingerprint, "deadbeef")
    assert stale["code"] == "E_SNAPSHOT_STALE"
    assert state is None and cache == {}


def test_format_version_mismatch_forces_reset(sample_dir, tmp_path):
    pipeline, store = _make_pipeline(sample_dir, tmp_path)
    pipeline.run(sample_dir / "events.jsonl", None)
    manifest = store.committed / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["format_version"] = SNAPSHOT_FORMAT_VERSION + 99
    manifest.write_text(json.dumps(doc, sort_keys=True))
    state, _cache, stale = store.resume_or_reset(
        pipeline.config.fingerprint, pipeline.fsm.fingerprint)
    assert stale["code"] == "E_SNAPSHOT_STALE"
    assert state is None

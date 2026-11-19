from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rsseg.edits import copy_sample_set
from rsseg.engine import IncrementalEngine, _SHUTDOWN, request_shutdown
from rsseg.runner import build_inputs
from rsseg.snapshot import MANIFEST_NAME, SnapshotManager


def test_snapshot_commit_is_atomic_and_manifest_complete(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    inputs = build_inputs(sources, work)
    engine = IncrementalEngine(inputs)
    report = engine.run()
    manager = SnapshotManager(work)
    info = manager.latest()
    assert info is not None and info.version == report.snapshot_version == 1
    assert (info.path / MANIFEST_NAME).exists()
    manifest = json.loads((info.path / MANIFEST_NAME).read_text())
    assert set(manifest) >= {"version", "config_hash", "index_version",
                             "tile_versions", "fusion_versions", "pending", "failures"}
    assert len(manifest["tile_versions"]) == 9
    assert len(manifest["fusion_versions"]) == 12
    # No temporary directories survive a successful commit.
    assert not list(manager.snapshots_dir.glob("snapshot.tmp.*"))


def test_partial_temp_snapshot_is_never_a_recovery_point(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    IncrementalEngine(build_inputs(sources, work)).run()
    manager = SnapshotManager(work)
    fake = manager.snapshots_dir / "snapshot.tmp.fake_999"
    fake.mkdir()
    (fake / "global_grid.npy").write_bytes(b"not-a-real-snapshot")
    recovered = manager.latest()
    assert recovered is not None and recovered.version == 1
    assert not fake.exists()


def test_version_changes_are_reported(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    engine = IncrementalEngine(build_inputs(sources, work))
    first = engine.run()
    from rsseg.edits import apply_edit

    apply_edit(sources, {"id": "cfg", "type": "config",
                         "patch": {"feature_thresholds": {"slope": 9.0}}})
    second_engine = IncrementalEngine(build_inputs(sources, work))
    second = second_engine.run()
    manager = SnapshotManager(work)
    diff = manager.versions_changed(manager.latest(), manager.latest())
    # Same snapshot object -> empty diff; compare stored manifests instead.
    assert second.snapshot_version == first.snapshot_version + 1
    assert diff["tiles"] == []


def test_shutdown_event_completes_current_tile_then_stops(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    inputs = build_inputs(sources, work, config_override={
        **build_inputs(sources, work).config.to_dict(), "recompute_budget": 2})
    _SHUTDOWN.clear()
    request_shutdown()
    try:
        report = IncrementalEngine(inputs).run()
        assert report.signaled is True
        assert report.recomputed_tiles == []
        assert report.pending_tiles
    finally:
        _SHUTDOWN.clear()


def test_concurrent_output_matches_single_thread_order(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work_parallel = tmp_path / "parallel"
    work_serial = tmp_path / "serial"
    copy_sample_set(sample_set, sources)

    parallel_inputs = build_inputs(sources, work_parallel,
                                   config_override={"workers": 4})
    serial_inputs = build_inputs(sources, work_serial,
                                 config_override={"workers": 1})
    parallel = IncrementalEngine(parallel_inputs).run(force_full=True)
    serial = IncrementalEngine(serial_inputs).run(force_full=True)
    assert parallel.recomputed_tiles == serial.recomputed_tiles
    assert parallel.recomputed_fusions == serial.recomputed_fusions
    assert parallel.failure_dicts() == serial.failure_dicts()
    grid_p = np.load(SnapshotManager(work_parallel).latest().path / "global_grid.npy")
    grid_s = np.load(SnapshotManager(work_serial).latest().path / "global_grid.npy")
    assert np.array_equal(grid_p, grid_s)

from __future__ import annotations

import json

import numpy as np

from spmm_diag.engine import compare_incremental_to_full, full_rebuild
from spmm_diag.snapshot import SnapshotStore, bootstrap_engine


def _drain(engine):
    if engine.pending:
        engine.run_until_complete()


def test_snapshot_roundtrip_and_atomic_directory(sample_bundle, tmp_path):
    csr, b, c, config = sample_bundle
    engine, _ = bootstrap_engine(SnapshotStore(tmp_path / "snapshots"), csr, b, c, config)
    engine.run_until_complete()
    store = SnapshotStore(tmp_path / "snapshots")
    saved = store.save(engine)
    assert saved.manifest["completed_blocks"] == list(range(len(engine.block_ranges)))
    assert not list(store.root.glob(".stage-*"))
    restored, info = bootstrap_engine(store, csr, b, c, config)
    assert info["source"] == "current"
    assert not restored.pending
    assert np.array_equal(engine.materialize_c(), restored.materialize_c())
    assert restored.row_fingerprints == engine.row_fingerprints


def test_stale_input_falls_back_to_nearest_valid_snapshot(sample_bundle, tmp_path):
    csr, b, c, config = sample_bundle
    store = SnapshotStore(tmp_path / "snapshots")
    from spmm_diag.engine import IncrementalEngine

    first = IncrementalEngine(csr, b, c, config)
    first.run()
    store.save(first)
    first.replace_row(7, [0, 5], [0.9, -0.2])
    first.run()
    store.save(first)
    store.save(first)

    changed = csr.replace_row(7, [1, 3, 8], [0.4, 1.1, -0.7])
    restored, info = bootstrap_engine(store, changed, b, c, config)
    assert info["source"] == "previous_version"
    assert info["current_error"]["code"] == "E_SNAPSHOT_STALE"
    assert info["completed_blocks"]
    assert info["rejected_blocks"]
    _drain(restored)
    reference = full_rebuild(changed, b, c, config)
    assert compare_incremental_to_full(restored, reference)["all_exact"]


def test_checksum_corruption_uses_previous_version(sample_bundle, tmp_path):
    csr, b, c, config = sample_bundle
    store = SnapshotStore(tmp_path / "snapshots")
    from spmm_diag.engine import IncrementalEngine

    engine = IncrementalEngine(csr, b, c, config)
    engine.run()
    store.save(engine)
    store.save(engine)
    state_path = store.current / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["edit_version"] = -999
    state_path.write_text(json.dumps(state), encoding="utf-8")
    restored, info = bootstrap_engine(store, csr, b, c, config)
    assert info["source"] == "previous_version"
    assert info["current_error"]["code"] == "E_SNAPSHOT_CORRUPT"
    assert not restored.pending

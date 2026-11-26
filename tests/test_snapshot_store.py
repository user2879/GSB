from __future__ import annotations

import pytest

from reco_shard_diag.persistence import (
    SnapshotStore,
    SnapshotVersionMismatch,
    termination_classification,
)


def test_snapshot_commit_pointer_and_version_mismatch(tmp_path):
    store = SnapshotStore(tmp_path, global_version=1, base_config_hash="h1")
    assert store.load() is None
    manifest = store.commit({"phase": "RUNNING", "completed_cursor": ["s1"]})
    loaded = store.load()
    assert loaded["phase"] == "RUNNING"
    assert (
        termination_classification(loaded)["kind"]
        == "RECOVERED_AFTER_FORCED_TERMINATION"
    )
    assert manifest["snapshot_dir"].startswith("snapshots/")
    with pytest.raises(SnapshotVersionMismatch):
        SnapshotStore(tmp_path, global_version=2, base_config_hash="h1").load()
    with pytest.raises(SnapshotVersionMismatch):
        SnapshotStore(tmp_path, global_version=1, base_config_hash="h2").load()

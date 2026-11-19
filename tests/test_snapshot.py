import json
import os

import pytest

from probegi.errors import SnapshotStaleError
from probegi.snapshot import SnapshotStore


def test_commit_and_load_roundtrip(tmp_path):
    store = SnapshotStore(str(tmp_path))
    m = store.commit({"a/data.json": {"x": 1}, "b.bin": b"\x00\x01"})
    assert m["version"] == 1
    state = store.load()
    assert state["version"] == 1
    assert json.loads(state["files"]["a/data.json"]) == {"x": 1}
    assert state["files"]["b.bin"] == b"\x00\x01"


def test_stale_base_version_rejected(tmp_path):
    store = SnapshotStore(str(tmp_path))
    store.commit({"f": b"1"})
    store.commit({"f": b"2"})
    with pytest.raises(SnapshotStaleError) as exc:
        store.commit({"f": b"3"}, base_version=1)
    assert exc.value.code == "E_SNAPSHOT_STALE"


def test_identical_content_rejected(tmp_path):
    store = SnapshotStore(str(tmp_path))
    store.commit({"f": b"same"})
    with pytest.raises(SnapshotStaleError):
        store.commit({"f": b"same"})


def test_orphan_partial_reclaimed_on_recovery(tmp_path):
    store = SnapshotStore(str(tmp_path))
    store.commit({"f": b"committed"})
    # Simulate a crashed commit: an unpromoted .partial directory.
    partial = os.path.join(str(tmp_path), "snapshots", "v000002.partial")
    os.makedirs(partial)
    with open(os.path.join(partial, "junk.bin"), "wb") as fh:
        fh.write(b"uncommitted")
    store2 = SnapshotStore(str(tmp_path))  # recovery runs in __init__
    assert not os.path.exists(partial)
    assert os.path.exists(os.path.join(str(tmp_path), "orphans",
                                       "v000002.partial", "junk.bin"))
    state = store2.load()
    assert state["version"] == 1  # last committed state only
    log = (tmp_path / "recovery.log").read_text()
    assert "orphan_reclaimed" in log


def test_corrupted_snapshot_detected(tmp_path):
    store = SnapshotStore(str(tmp_path))
    store.commit({"f.bin": b"original"})
    path = tmp_path / "snapshots" / "v000001" / "f.bin"
    path.write_bytes(b"tampered")
    with pytest.raises(SnapshotStaleError):
        store.load()


def test_no_snapshot_load_raises(tmp_path):
    store = SnapshotStore(str(tmp_path))
    with pytest.raises(SnapshotStaleError):
        store.load()

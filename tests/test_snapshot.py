from __future__ import annotations

import pytest

from ckptmeta.catalog import Catalog
from ckptmeta.errors import InteropError
from ckptmeta.snapshot import SnapshotStore


def test_snapshot_save_restore_and_replay(fixtures, loaded, tmp_path):
    config, ledger = loaded
    catalog = Catalog(fixtures.direct, config, ledger)
    with SnapshotStore(tmp_path / "snapshots.sqlite3") as store:
        snapshot = store.save(catalog)
        assert snapshot.format_versions == [3]
        assert len(snapshot.items) == 3
        restored = store.restore(catalog)
        assert restored["matched"] is True
        assert restored["replay"]["reproduced"] is True
        replay = store.replay(catalog)
        assert replay["compared_item_count"] == 3


def test_snapshot_mismatch_replaces_state_and_preserves_old_snapshot_readonly(fixtures, loaded, tmp_path):
    config, ledger = loaded
    catalog = Catalog(fixtures.direct, config, ledger)
    db = tmp_path / "snapshots.sqlite3"
    with SnapshotStore(db) as store:
        old = store.save(catalog)
        fixtures.direct_target.write_text(
            fixtures.direct_target.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        restored = store.restore(catalog)
        assert restored["matched"] is False
        assert restored["replaced"] is True
        assert restored["old_snapshot_readonly"] is True
        assert restored["replay"]["reproduced"] is True
        listed = store.list()
        old_row = next(row for row in listed if row["id"] == old.id)
        assert old_row["readonly"] == 1
        assert old_row["active"] == 0
        with pytest.raises(InteropError) as raised:
            store.replay(catalog, snapshot_id=old.id)
        assert raised.value.code == "E_REPLAY"
        assert raised.value.details["reason"] == "content_digest"


def test_snapshot_database_has_required_fields(fixtures, loaded, tmp_path):
    config, ledger = loaded
    catalog = Catalog(fixtures.direct, config, ledger)
    db = tmp_path / "snapshots.sqlite3"
    with SnapshotStore(db) as store:
        snapshot = store.save(catalog)
        state = snapshot.state
        assert state["format_versions"] == [3]
        assert state["content_digest"]
        assert state["normalized_view_digest"]
        assert state["upgrade_trace"] == []
        assert state["processed_cursor"] == 0
        rows = store.connection.execute(
            "SELECT format_versions_json, content_digest, normalized_view_digest, state_json FROM snapshots WHERE id = ?",
            (snapshot.id,),
        ).fetchone()
        assert rows is not None
        assert '"3"' not in rows["format_versions_json"] or "[3]" in rows["format_versions_json"]

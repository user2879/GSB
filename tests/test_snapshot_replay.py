import json

import pytest

from cfgdist.distributor import distribute
from cfgdist.errors import BizError, ErrorCode
from cfgdist.manifest import load_manifest, manifest_to_json
from cfgdist.planner import build_plan
from cfgdist.replay import replay_latest
from cfgdist.runtime import NodePool, load_node_declaration
from cfgdist.session import SessionStore
from cfgdist.snapshot import build_snapshot, load_snapshot, restore_snapshot, save_snapshot


def _run_happy_path(tmp_path, v3_manifest, node_specs):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_to_json(v3_manifest), encoding="utf-8")
    nodes_path = tmp_path / "nodes.json"
    nodes_path.write_text(json.dumps({"nodes": node_specs}), encoding="utf-8")
    plan = build_plan(v3_manifest, [spec["id"] for spec in node_specs])
    db = tmp_path / "sessions.db"
    with NodePool(node_specs):
        with SessionStore(db) as store:
            report = distribute(plan, node_specs, store)
            snapshot = build_snapshot(v3_manifest, plan, store, report["session_id"])
    snapshot_path = tmp_path / "snapshot.json"
    save_snapshot(snapshot, snapshot_path)
    return manifest_path, nodes_path, db, snapshot_path, plan


def test_replay_recomputes_and_matches_committed_session(
    tmp_path, v3_manifest, node_specs
):
    manifest_path, nodes_path, db, _, _ = _run_happy_path(tmp_path, v3_manifest, node_specs)
    report = replay_latest(db, manifest_path, nodes_path)
    assert report["ok"] is True
    assert set(report["verified_nodes"]) == {"node-a", "node-b", "node-c"}


def test_replay_detects_missing_config_version(tmp_path, v3_manifest, node_specs):
    manifest_path, nodes_path, db, _, _ = _run_happy_path(tmp_path, v3_manifest, node_specs)
    changed = json.loads(manifest_path.read_text())
    for entry in changed["entries"]:
        entry["config_version"] = 99
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(BizError) as exc_info:
        replay_latest(db, manifest_path, nodes_path)
    assert exc_info.value.code == ErrorCode.E_REPLAY
    assert exc_info.value.details["missing_config_version"] == 3


def test_replay_detects_node_downgrade(tmp_path, v3_manifest, node_specs):
    manifest_path, nodes_path, db, _, _ = _run_happy_path(tmp_path, v3_manifest, node_specs)
    downgraded = [spec for spec in node_specs if spec["id"] != "node-c"]
    nodes_path.write_text(json.dumps({"nodes": downgraded}), encoding="utf-8")
    with pytest.raises(BizError) as exc_info:
        replay_latest(db, manifest_path, nodes_path)
    assert exc_info.value.code == ErrorCode.E_REPLAY
    assert exc_info.value.details["downgraded_nodes"] == ["node-c"]


def test_replay_locates_first_digest_diff(tmp_path, v3_manifest, node_specs):
    manifest_path, nodes_path, db, _, plan = _run_happy_path(
        tmp_path, v3_manifest, node_specs
    )
    # Tamper with a stored receipt digest inside the committed session.
    with SessionStore(db) as store:
        committed = store.latest_committed_version()
        session_id = committed["session_id"]
        row = store.conn.execute(
            "SELECT raw_json FROM receipts WHERE session_id = ? AND node = ?",
            (session_id, "node-a"),
        ).fetchone()
        raw = json.loads(row["raw_json"])
        raw["digest"] = "deadbeef"
        store.conn.execute(
            "UPDATE receipts SET digest = ?, raw_json = ? WHERE session_id = ? AND node = ?",
            ("deadbeef", json.dumps(raw, sort_keys=True), session_id, "node-a"),
        )
        store.conn.commit()
    with pytest.raises(BizError) as exc_info:
        replay_latest(db, manifest_path, nodes_path)
    assert exc_info.value.code == ErrorCode.E_REPLAY
    first_diff = exc_info.value.details["first_diff"]
    assert first_diff["node"] == "node-a"
    assert first_diff["field"] == "receipt.digest"


def test_snapshot_conflict_does_full_replacement_and_archives_readonly(
    tmp_path, v3_manifest, node_specs
):
    manifest_path, nodes_path, db, snapshot_path, _ = _run_happy_path(
        tmp_path, v3_manifest, node_specs
    )
    snapshot = load_snapshot(snapshot_path)
    snapshot["manifest_version"] = 2  # force a version conflict
    conflicting = tmp_path / "conflict.json"
    save_snapshot(snapshot, conflicting)

    current_manifest = load_manifest(manifest_path)
    archive_dir = tmp_path / "archive"
    with SessionStore(db) as store:
        with pytest.raises(BizError) as exc_info:
            restore_snapshot(
                conflicting,
                current_manifest,
                store,
                session_id="restored-1",
                archive_dir=archive_dir,
            )
    assert exc_info.value.code == ErrorCode.E_VERSION
    assert exc_info.value.details["replaced"] is True
    assert exc_info.value.details["mode"] == "full_replacement"

    archived = list(archive_dir.glob("*.json"))
    assert len(archived) == 2
    import os

    assert all(not os.access(path, os.W_OK) for path in archived)

    # The store was fully replaced: only the restored session remains.
    with SessionStore(db) as store:
        rows = store.conn.execute("SELECT session_id FROM sessions").fetchall()
        assert [row["session_id"] for row in rows] == ["restored-1"]
        restored = store.get_session("restored-1")
        assert restored["manifest_version"] == 2


def test_compatible_snapshot_restores_without_error(
    tmp_path, v3_manifest, node_specs
):
    manifest_path, _, db, snapshot_path, _ = _run_happy_path(
        tmp_path, v3_manifest, node_specs
    )
    current_manifest = load_manifest(manifest_path)
    with SessionStore(db) as store:
        report = restore_snapshot(
            snapshot_path,
            current_manifest,
            store,
            session_id="restored-ok",
            archive_dir=tmp_path / "archive",
        )
    assert report["replaced"] is False
    assert report["mode"] == "compatible_restore"

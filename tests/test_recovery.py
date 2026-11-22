import json
import sqlite3

import pytest

from edge_lease.errors import ReplayError, SnapshotError
from edge_lease.ledger import FORMAT_VERSION, Ledger
from edge_lease.manifest import load_manifests, load_registry, load_shards
from edge_lease.pipeline import PipelineConfig, run_pipeline
from edge_lease.pipeline import SimulatedHardKill
from edge_lease.replay import reopen, replay


def _config(workspace, **overrides):
    defaults = dict(
        workdir=workspace / "run",
        registry_path=workspace / "nodes.json",
        shards_path=workspace / "shards.json",
        manifest_path=workspace / "manifest.json",
        scale=1000,
        content_seed=20261123,
        global_limit=3,
        batch_size=3,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _load_manifests(workspace):
    registry = load_registry(workspace / "nodes.json")
    shards = load_shards(workspace / "shards.json")
    return load_manifests(workspace / "manifest.json", registry, shards)


def test_replay_reproduces_every_committed_receipt(sample_workspace):
    run_pipeline(_config(sample_workspace))
    ledger = Ledger(sample_workspace / "run" / "lease.sqlite")
    try:
        manifests = _load_manifests(sample_workspace)
        report = replay(ledger, manifests, 20261123, 1000)
        assert report.unknown_batches == []
        assert report.rerun_required == []
        assert report.replayed_items == report.matched_items
        assert report.replayed_items > 0
        assert all(row["ok"] for row in report.rows)
    finally:
        ledger.close()


def test_hard_kill_leaves_unknown_batch_requiring_rerun(sample_workspace):
    run_pipeline(_config(sample_workspace))
    db_path = sample_workspace / "run" / "lease.sqlite"
    # Simulate a kill after begin_batch but before commit_batch: insert a raw
    # uncommitted batch row directly into the ledger.
    raw = sqlite3.connect(db_path)
    raw.execute(
        "INSERT INTO batches(batch_id, version, state, committed, total_items,"
        " ok_items, quota_json, created_ms) VALUES('v9-b7', 1, 'UNKNOWN', 0,"
        " 2, 0, '{}', 12345)"
    )
    raw.commit()
    raw.close()
    ledger = Ledger(db_path)
    try:
        report = reopen(ledger)
        assert [batch["batch_id"] for batch in report.unknown_batches] == ["v9-b7"]
        assert report.rerun_required == ["v9-b7"]
        # rerun must never be reported as committed
        rows = ledger.unknown_batches()
        assert rows and rows[0]["state"] == "RERUN_REQUIRED"
    finally:
        ledger.close()


def test_replay_detects_first_inconsistent_field(sample_workspace):
    run_pipeline(_config(sample_workspace))
    db_path = sample_workspace / "run" / "lease.sqlite"
    # Corrupt one committed receipt digest -> replay must locate path/node/field
    raw = sqlite3.connect(db_path)
    row = raw.execute(
        "SELECT id, receipt_json FROM receipts ORDER BY id ASC LIMIT 1"
    ).fetchone()
    doc = json.loads(row[1])
    doc["derived_digest"] = doc["derived_digest"] + 1
    raw.execute(
        "UPDATE receipts SET receipt_json=? WHERE id=?",
        (json.dumps(doc, sort_keys=True), row[0]),
    )
    raw.commit()
    raw.close()
    ledger = Ledger(db_path)
    try:
        manifests = _load_manifests(sample_workspace)
        with pytest.raises(ReplayError) as exc:
            replay(ledger, manifests, 20261123, 1000)
        assert exc.value.code == "E_REPLAY"
        assert doc["path"] in exc.value.location
        assert doc["node_id"] in exc.value.location
        assert "derived_digest" in exc.value.location
    finally:
        ledger.close()


def test_snapshot_whole_state_replacement_on_identity_drift(sample_workspace):
    run_pipeline(_config(sample_workspace))
    db_path = sample_workspace / "run" / "lease.sqlite"
    raw = sqlite3.connect(db_path)
    identity = json.loads(
        raw.execute("SELECT value FROM meta WHERE key='session_identity'").fetchone()[0]
    )
    before_plans = raw.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    before_batches = raw.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    # Same version but tampered manifest digest => forced wholesale replacement
    identity["manifest_digest"] = "sha-fnv64:deadbeef"
    raw.execute(
        "UPDATE meta SET value=? WHERE key='session_identity'",
        (json.dumps(identity, sort_keys=True),),
    )
    raw.commit()
    raw.close()

    from edge_lease.logging_setup import JsonlLogger
    from edge_lease.session import Session

    logger = JsonlLogger(sample_workspace / "run" / "events2.jsonl")
    ledger = Ledger(db_path, logger=logger)
    try:
        manifests = _load_manifests(sample_workspace)
        session = Session(ledger, logger, 20261123, 1000)
        diff = session.check_or_replace_snapshot(manifests[-1])
        assert diff.replaced is True
        assert diff.reason == "snapshot_identity_mismatch"
        assert diff.before["manifest_digest"] == "sha-fnv64:deadbeef"
        # all runtime rows were replaced (deleted), identity rewritten
        assert ledger.conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
        assert ledger.conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0
        assert len(diff.changed_items) >= before_plans and before_batches >= 1
    finally:
        ledger.close()


def test_unsupported_format_version_raises_snapshot_error(sample_workspace):
    run_pipeline(_config(sample_workspace))
    db_path = sample_workspace / "run" / "lease.sqlite"
    raw = sqlite3.connect(db_path)
    raw.execute(
        "UPDATE meta SET value='999' WHERE key='format_version'"
    )
    raw.commit()
    raw.close()
    with pytest.raises(SnapshotError) as exc:
        Ledger(db_path)
    assert exc.value.code == "E_SNAPSHOT"


def test_state_machine_history_is_persisted(sample_workspace):
    run_pipeline(_config(sample_workspace))
    ledger = Ledger(sample_workspace / "run" / "lease.sqlite")
    try:
        history = ledger.state_history()
        transitions = [(row["from_state"], row["to_state"]) for row in history]
        assert transitions[0] == ("INIT", "LOADED")
        assert ("LOADED", "COMPILED") in transitions
        assert any(to_state == "CONVERGED" for _from, to_state in transitions)
        assert FORMAT_VERSION == 1
    finally:
        ledger.close()


def test_hard_kill_then_rerun_and_replay_is_consistent(sample_workspace):
    config = _config(sample_workspace, batch_size=2, crash_after_batches=1)
    with pytest.raises(SimulatedHardKill):
        run_pipeline(config)
    db_path = sample_workspace / "run" / "lease.sqlite"

    # Reopen: only committed batches are trusted; nothing is fabricated.
    ledger = Ledger(db_path)
    try:
        report = reopen(ledger)
        assert report.committed_batches, "the killed batch was committed first"
        assert report.unknown_batches == []
    finally:
        ledger.close()

    # Rerun the same command: whole-state identity matches, work proceeds and
    # every item converges.
    rerun_config = _config(sample_workspace, batch_size=2)
    result = run_pipeline(rerun_config)
    assert all(
        all(row["ok"] for row in version["convergence_rows"])
        for version in result["versions"]
    )

    ledger = Ledger(db_path)
    try:
        manifests = _load_manifests(sample_workspace)
        replay_report = replay(ledger, manifests, 20261123, 1000)
        assert replay_report.matched_items == replay_report.replayed_items
        assert replay_report.unknown_batches == []
    finally:
        ledger.close()

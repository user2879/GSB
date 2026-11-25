from __future__ import annotations

import sqlite3

import pytest

from lease_arbiter.canonical import canonical_json, digest_text
from lease_arbiter.engine import LeaseArbiterEngine
from lease_arbiter.errors import ReplayError
from lease_arbiter.ledger import Ledger
from lease_arbiter.recovery import replay_against_ledger

from .conftest import clone_declaration, write_local_spec


def _commit_first(spec, ledger: Ledger) -> None:
    engine = LeaseArbiterEngine(spec)
    event = spec.events[0]
    expiry = engine.expire_due(event.at_ms, event.seq)
    applied = engine.apply_event(event)
    ledger.commit_event(
        event_seq=event.seq,
        outcome=applied.outcome,
        reclaims=expiry.reclaims + applied.reclaims,
        transitions=expiry.transitions + applied.transitions,
        ownership_changes=expiry.ownership_changes + applied.ownership_changes,
        snapshot=engine.snapshot(),
    )


def test_snapshot_replacement_is_whole_not_incremental(tmp_path):
    spec = write_local_spec(tmp_path)
    path = tmp_path / "ledger.sqlite3"
    ledger = Ledger(path, spec, LeaseArbiterEngine(spec).snapshot(), fresh=True)
    _commit_first(spec, ledger)
    assert len(ledger.read_commits()) == 1
    ledger.close()

    spec2 = clone_declaration(tmp_path, spec, declaration_version=2)
    ledger2 = Ledger(path, spec2, LeaseArbiterEngine(spec2).snapshot(), fresh=False)
    try:
        assert ledger2.snapshot_replaced is True
        assert ledger2.replacement_reason == "declaration_version_changed"
        assert ledger2.read_commits() == []
        assert ledger2.load_snapshot()["last_event_seq"] == 0
        session = ledger2.session_info()
        assert session["declaration_version"] == spec2.declaration_version
        assert session["snapshot_version"] == 1
        with sqlite3.connect(path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM session_replacements").fetchone()[0]
        assert count == 2
    finally:
        ledger2.close()


def test_pending_request_is_marked_unknown_and_slots_are_reported_released(tmp_path):
    spec = write_local_spec(tmp_path)
    path = tmp_path / "pending.sqlite3"
    ledger = Ledger(path, spec, LeaseArbiterEngine(spec).snapshot(), fresh=True)
    ledger.record_pending(
        1,
        {
            "event_seq": 1,
            "reservations": [
                {"lease_id": "a-cache-1", "node_id": "alpha", "global_slot": 1, "node_slot": 1}
            ],
            "actions": [],
        },
        request_id="pending-1",
    )
    ledger.close()

    reopened = Ledger(path, spec, LeaseArbiterEngine(spec).snapshot(), fresh=False)
    try:
        recovered = reopened.recover_pending()
        assert len(recovered) == 1
        assert recovered[0]["status"] == "unknown"
        assert recovered[0]["released_global_slots"] == [1]
        assert recovered[0]["released_node_slots"] == [["alpha", 1]]
        assert reopened.pending_requests()[0]["status"] == "unknown"
        assert reopened.load_snapshot()["last_event_seq"] == 0
        assert len(reopened.recover_pending()) == 1
    finally:
        reopened.close()


def test_replay_reports_first_divergent_event_and_field(tmp_path):
    spec = write_local_spec(tmp_path)
    path = tmp_path / "tampered.sqlite3"
    ledger = Ledger(path, spec, LeaseArbiterEngine(spec).snapshot(), fresh=True)
    _commit_first(spec, ledger)
    ledger.close()

    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT payload_json FROM commits WHERE event_seq = 1").fetchone()
        payload = __import__("json").loads(row[0])
        payload["outcome"]["status"] = "E_QUOTA"
        payload_json = canonical_json(payload)
        conn.execute(
            "UPDATE commits SET payload_json = ?, content_digest = ? WHERE event_seq = 1",
            (payload_json, digest_text(payload_json)),
        )
        conn.commit()

    with pytest.raises(ReplayError) as caught:
        replay_against_ledger(spec, path)
    error = caught.value
    assert error.code == "E_REPLAY"
    assert error.details["event_seq"] == 1
    assert error.details["field"] == "status"
    assert error.details["expected"] == "granted"
    assert error.details["actual"] == "E_QUOTA"

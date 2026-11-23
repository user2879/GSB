from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from leaseaudit.clock import ManualClock
from leaseaudit.engine import LeaseEngine
from leaseaudit.errors import BusinessError, ErrorCode
from leaseaudit.ledger import GENESIS_DIGEST, AuditLedger
from leaseaudit.replay import replay_ledger
from leaseaudit.scenario import validate_scenario
from leaseaudit.util import canonical_json, sha256_text
from leaseaudit.view import build_audit_view


def _make_ledger(path: Path, scenario):
    ledger = AuditLedger(path, scenario=scenario, verify=True)
    clock = ManualClock(value_ms=8_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    lease = engine.issue("alpha", 2)
    clock.advance(100)
    engine.renew(lease.lease_id, "alpha")
    engine.issue("beta", 1)
    engine.observe_all(force=True)
    engine.release(lease.lease_id)
    engine.close()
    return path


def test_clean_replay_and_stable_read_only_view(tmp_path, scenario):
    path = _make_ledger(tmp_path / "audit.sqlite", scenario)
    result = replay_ledger(path, scenario)
    assert result["status"] == "OK"
    assert result["ledger_defect"] is False
    view_one = build_audit_view(path)
    view_two = build_audit_view(path)
    assert view_one == view_two
    assert [event["seq"] for event in view_one["events"]] == sorted(
        event["seq"] for event in view_one["events"]
    )
    assert path.stat().st_size > 0


def test_external_drift_is_separate_from_ledger_defect(tmp_path, scenario, scenario_doc):
    path = _make_ledger(tmp_path / "audit.sqlite", scenario)
    drifted_doc = json.loads(json.dumps(scenario_doc))
    drifted_doc["scenario_version"] = "drifted-v2"
    drifted_doc["sources"][0]["port"] = 55999
    drifted_doc["sources"][1]["processing_ms"] = 44
    drifted = validate_scenario(drifted_doc)
    result = replay_ledger(path, drifted)
    assert result["status"] == "EXTERNAL_DRIFT"
    assert result["business_code"] == "E_EXTERNAL_DRIFT"
    assert result["ledger_defect"] is False
    assert result["external_drift"]["classification"] == "EXTERNAL_DRIFT"
    changed_fields = {
        difference["field_path"] for difference in result["external_drift"]["field_differences"]
    }
    assert "scenario_version" in changed_fields
    assert any("sources" in field and "port" in field for field in changed_fields)


def _resign_ledger(path: Path, seq: int, mutator) -> None:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM events WHERE seq=?", (seq,)).fetchone()
        payload = json.loads(row["payload_json"])
        mutator(payload)
        conn.execute(
            "UPDATE events SET payload_json=? WHERE seq=?",
            (canonical_json(payload), seq),
        )
        digest = GENESIS_DIGEST
        for event in conn.execute("SELECT * FROM events ORDER BY seq ASC"):
            event_payload = json.loads(event["payload_json"])
            event_hash = AuditLedger._event_hash(
                event["seq"],
                event["event_id"],
                event_type=event["event_type"],
                source_id=event["source_id"],
                lease_id=event["lease_id"],
                amount=event["amount"],
                mono_ms=event["mono_ms"],
                params_version=event["params_version"],
                params_digest=event["params_digest"],
                progress_cursor=event["progress_cursor"],
                payload=event_payload,
            )
            conn.execute(
                "UPDATE events SET previous_digest=?, event_hash=? WHERE seq=?",
                (digest, event_hash, event["seq"]),
            )
            digest = sha256_text(digest + event_hash)
        conn.execute("UPDATE ledger_meta SET content_digest=? WHERE id=1", (digest,))
        conn.commit()
    finally:
        conn.close()


def test_replay_reports_first_mismatch_sequence_source_and_field(tmp_path, scenario):
    path = _make_ledger(tmp_path / "audit.sqlite", scenario)

    def corrupt(payload):
        payload["global_allocatable_after"] += 5

    _resign_ledger(path, 1, corrupt)
    AuditLedger(path, readonly=True, verify=True).close()
    with pytest.raises(BusinessError) as caught:
        replay_ledger(path, scenario)
    error = caught.value
    assert error.code == ErrorCode.E_REPLAY
    assert error.event_seq == 1
    assert error.source_id == "alpha"
    assert error.field_path == "global_allocatable_after"
    assert error.details["expected"] == 8
    assert error.details["actual"] == 13

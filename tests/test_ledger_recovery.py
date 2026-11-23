from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from leaseaudit.clock import ManualClock, RealMonotonicClock
from leaseaudit.engine import LeaseEngine
from leaseaudit.errors import BusinessError, ErrorCode
from leaseaudit.ledger import AuditLedger
from leaseaudit.replay import replay_ledger
from leaseaudit.scenario import validate_scenario
from leaseaudit.util import read_json_strict, write_json_atomic


def _build_ledger(path: Path, scenario):
    ledger = AuditLedger(path, scenario=scenario, verify=True)
    clock = ManualClock(value_ms=6_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    lease = engine.issue("alpha", 2)
    engine.renew(lease.lease_id, "alpha")
    return engine


def test_content_digest_tamper_is_esnapshot(tmp_path, scenario):
    path = tmp_path / "audit.sqlite"
    engine = _build_ledger(path, scenario)
    engine.close()
    conn = sqlite3.connect(path)
    try:
        payload_json = conn.execute("SELECT payload_json FROM events WHERE seq=1").fetchone()[0]
        payload = json.loads(payload_json)
        payload["tamper"] = True
        conn.execute(
            "UPDATE events SET payload_json=? WHERE seq=1",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(BusinessError) as caught:
        AuditLedger(path, readonly=True, verify=True)
    assert caught.value.code == ErrorCode.E_SNAPSHOT
    assert caught.value.field_path == "$.events[1].event_hash"
    assert caught.value.details["expected"] != caught.value.details["actual"]


def test_snapshot_restore_whole_ledger(tmp_path, scenario):
    path = tmp_path / "audit.sqlite"
    engine = _build_ledger(path, scenario)
    manifest = engine.ledger.create_snapshot(tmp_path / "snapshots")
    engine.close()

    ledger = AuditLedger(path, scenario=scenario, verify=True)
    engine_two = LeaseEngine(scenario, ledger, clock=ManualClock(value_ms=6_100_000))
    try:
        engine_two.issue("gamma", 2)
        changed_digest = ledger.verify_integrity()["content_digest"]
    finally:
        engine_two.close()
    assert changed_digest != manifest["content_digest"]

    restored = AuditLedger.restore_snapshot(
        tmp_path / "snapshots" / manifest["snapshot_id"], path
    )
    assert restored["status"] == "RESTORED"
    assert restored["content_digest"] == manifest["content_digest"]
    assert replay_ledger(path, scenario)["status"] == "OK"


def test_pending_intent_becomes_unknown_and_never_claims_exit_report(tmp_path, scenario):
    path = tmp_path / "audit.sqlite"
    ledger = AuditLedger(path, scenario=scenario, verify=True)
    clock = ManualClock(value_ms=7_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    committed = engine.issue("alpha", 1)
    engine.close()

    class SimulatedCrash(Exception):
        pass

    ledger = AuditLedger(path, scenario=scenario, verify=True)
    crashing_engine = LeaseEngine(
        scenario,
        ledger,
        clock=ManualClock(value_ms=7_000_100),
        crash_after_intent=lambda _intent: (_ for _ in ()).throw(SimulatedCrash()),
    )
    with pytest.raises(SimulatedCrash):
        crashing_engine.issue("alpha", 1)
    crashing_engine.close()

    recovered_ledger = AuditLedger(path, scenario=scenario, verify=True)
    recovered = LeaseEngine(scenario, recovered_ledger, clock=ManualClock(value_ms=7_000_200))
    try:
        state = recovered.snapshot()
        unknown_events = [
            event for event in recovered_ledger.events() if event.event_type == "LEASE_UNKNOWN"
        ]
    finally:
        recovered.close()
    assert state["exit_report_available"] is False
    assert len(state["unknown_leases"]) == 1
    assert len(unknown_events) == 1
    assert state["sources"][0]["active_leases"][0]["lease_id"] == committed.lease_id
    assert replay_ledger(path, scenario)["status"] == "OK"


def test_snapshot_version_mismatch_never_replaces_target(tmp_path, scenario):
    path = tmp_path / "audit.sqlite"
    engine = _build_ledger(path, scenario)
    manifest = engine.ledger.create_snapshot(tmp_path / "snapshots")
    target_digest = engine.ledger.verify_integrity()["content_digest"]
    engine.close()
    snapshot_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
    bad_manifest = read_json_strict(snapshot_dir / "manifest.json")
    bad_manifest["ledger_format_version"] = 999
    write_json_atomic(snapshot_dir / "manifest.json", bad_manifest)
    with pytest.raises(BusinessError) as caught:
        AuditLedger.restore_snapshot(snapshot_dir, path)
    assert caught.value.code == ErrorCode.E_SNAPSHOT
    assert caught.value.field_path == "$.snapshot.manifest.ledger_format_version"
    unchanged = AuditLedger(path, readonly=True, verify=True)
    try:
        assert unchanged.verify_integrity()["content_digest"] == target_digest
    finally:
        unchanged.close()


def test_existing_ledger_rejects_different_scenario(tmp_path, scenario_doc, scenario):
    path = tmp_path / "audit.sqlite"
    ledger = AuditLedger(path, scenario=scenario, verify=True)
    ledger.close()
    changed = json.loads(json.dumps(scenario_doc))
    changed["scenario_version"] = "other-version"
    with pytest.raises(BusinessError) as caught:
        AuditLedger(path, scenario=validate_scenario(changed), verify=True)
    assert caught.value.code == ErrorCode.E_SNAPSHOT
    assert caught.value.field_path == "$.ledger_meta.scenario_digest"

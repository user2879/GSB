from __future__ import annotations

import copy
import json

import pytest

from shard_diag.errors import DiagnosticError, ErrorCode
from shard_diag.ledger import Ledger, LEDGER_FORMAT_VERSION


def _lease(seq, db_id, shard, lease_id, version=1):
    return {
        "lease_id": lease_id,
        "db_id": db_id,
        "shard_id": shard,
        "event_seq": seq,
        "granted_at_ms": 100,
        "expires_at_ms": 200,
        "renew_count": 0,
        "lease_version": version,
        "route_version": 2,
        "status": "released",
        "terminal_at_ms": 150,
        "terminal_reason": "probe_complete",
    }


def _record(seq, db_id, table, key, key_column):
    return {
        "event_seq": seq,
        "db_id": db_id,
        "table": table,
        "key_column": key_column,
        "route_key": key,
        "old_route_version": 1,
        "new_route_version": 2,
        "old_shard": f"{table}:s0",
        "new_shard": f"{table}:s1",
        "changed": True,
        "drift_direction": "right",
        "boundary_index": 1,
        "boundary_old": 100,
        "boundary_new": 80,
        "boundary_offset": -20,
        "reason": "fixture drift",
    }


def _event(seq, db_id, table, key, key_column, lease):
    return {
        "event_seq": seq,
        "db_id": db_id,
        "table": table,
        "route_key": key,
        "hit_shards": [f"{table}:s1"],
        "sample_keys": [key],
        "worker_route_version": 2,
        "worker_elapsed_ms": 0.25,
        "leases": [lease],
        "thread_name": "shard-probe_0",
        "route_record": _record(seq, db_id, table, key, key_column),
    }


def _payload(spec, session_id, snapshot_digest, batch_index=1):
    event1 = _event(1, "db-a", "orders", 210, "order_id", _lease(1, "db-a", "orders:s1", "lease-1"))
    event2 = _event(2, "db-a", "audit", 45, "audit_id", _lease(2, "db-a", "audit:s1", "lease-2"))
    events = [event1, event2]
    records = [event1["route_record"], event2["route_record"]]
    return {
        "payload_format_version": 1,
        "session_id": session_id,
        "snapshot_digest": snapshot_digest,
        "batch_id": f"batch-{batch_index:03d}",
        "batch_index": batch_index,
        "start_seq": 1,
        "end_seq": 2,
        "events": events,
        "route_records": records,
        "drift_records": copy.deepcopy(records),
        "leases": [event1["leases"][0], event2["leases"][0]],
        "lease_transitions": [],
        "quota_snapshot": {
            "global_lease_limit": spec.global_lease_limit,
            "active_leases": 0,
            "remaining_global": spec.global_lease_limit,
            "by_db": {
                database.id: {
                    "active_leases": 0,
                    "lease_limit": database.lease_limit,
                    "remaining_leases": database.lease_limit,
                    "held_shards": [],
                }
                for database in spec.databases
            },
            "held_shards": [],
        },
        "wait_records": [],
        "high_water": {"max_active_global": 2, "max_inflight": 1},
        "probe_sample_count": 1,
        "progress": {"last_batch_index": batch_index, "last_seq": 2},
    }


def _committed_ledger(tmp_path, spec):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    snapshot = spec.snapshot(generation=1)
    installed = ledger.replace_snapshot("session-test", snapshot, generation=1)
    ledger.begin_batch(
        session_id="session-test",
        snapshot_digest=installed["snapshot_digest"],
        batch_id="batch-001",
        batch_index=1,
        event_seqs=[1, 2],
    )
    payload = _payload(spec, "session-test", installed["snapshot_digest"])
    committed = ledger.commit_batch(payload)
    return ledger, installed, payload, committed


def test_batch_commit_writes_all_required_rows_in_one_transaction(tmp_path, spec_bundle):
    ledger, installed, payload, committed = _committed_ledger(tmp_path, spec_bundle["spec"])
    try:
        assert committed["format_version"] == LEDGER_FORMAT_VERSION
        assert committed["content_digest"].startswith("sha256:")
        assert ledger.get_progress("session-test")["last_seq"] == 2
        counts = {}
        for table in ("responses", "routing_results", "drift_records", "leases", "ownership", "quota"):
            counts[table] = ledger.connection.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE session_id='session-test'"
            ).fetchone()["n"]
        assert counts["responses"] == 2
        assert counts["routing_results"] == 2
        assert counts["drift_records"] == 2
        assert counts["leases"] == 2
        assert counts["ownership"] == 2
        assert counts["quota"] == 3
        assert ledger.verify_integrity("session-test")["verified"]
        replay = ledger.replay_against("session-test", [payload])
        assert replay["verified"]
    finally:
        ledger.close()


def test_snapshot_replacement_is_whole_not_merge(tmp_path, spec_bundle):
    ledger, installed, payload, committed = _committed_ledger(tmp_path, spec_bundle["spec"])
    try:
        assert len(ledger.read_batches("session-test")) == 1
        second_snapshot = spec_bundle["spec"].snapshot(generation=2)
        replaced = ledger.replace_snapshot("session-test", second_snapshot, generation=2)
        assert replaced["generation"] == 2
        assert ledger.read_batches("session-test") == []
        assert ledger.get_progress("session-test") is None
        assert ledger.list_intents("session-test") == []
        snapshot = ledger.get_snapshot("session-test")
        assert snapshot["generation"] == 2
        assert snapshot["snapshot_digest"] != installed["snapshot_digest"]
        with pytest.raises(DiagnosticError) as caught:
            ledger.replace_snapshot("session-test", second_snapshot, generation=1)
        assert caught.value.code == ErrorCode.STATE.value
    finally:
        ledger.close()


def test_failed_commit_rolls_back_every_batch_row(tmp_path, spec_bundle):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    installed = ledger.replace_snapshot("session-test", spec_bundle["spec"].snapshot(generation=1), generation=1)
    ledger.begin_batch(
        session_id="session-test",
        snapshot_digest=installed["snapshot_digest"],
        batch_id="batch-001",
        batch_index=1,
        event_seqs=[1, 2],
    )
    malformed = _payload(spec_bundle["spec"], "session-test", installed["snapshot_digest"])
    malformed["route_records"][0].pop("event_seq")
    try:
        with pytest.raises(KeyError):
            ledger.commit_batch(malformed)
        assert ledger.read_batches("session-test") == []
        assert ledger.get_progress("session-test") is None
        assert ledger.list_intents("session-test")[0]["status"] == "pending"
        for table in ("responses", "routing_results", "drift_records", "leases", "ownership", "quota"):
            assert ledger.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == 0
    finally:
        ledger.close()


def test_recovery_marks_pending_unknown_and_surfaces_rerun_requirement(tmp_path, spec_bundle):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    installed = ledger.replace_snapshot("session-test", spec_bundle["spec"].snapshot(generation=1), generation=1)
    ledger.begin_batch(
        session_id="session-test",
        snapshot_digest=installed["snapshot_digest"],
        batch_id="batch-pending",
        batch_index=1,
        event_seqs=[1, 2],
    )
    try:
        report = ledger.recover_session("session-test")
        assert report.unknown_batch_ids == ("batch-pending",)
        assert report.unknown_event_seqs == (1, 2)
        assert report.reclaimed_event_seqs == (1, 2)
        assert report.exit_report_available is False
        assert report.next_batch_index == 1

        ledger.begin_batch(
            session_id="session-test",
            snapshot_digest=installed["snapshot_digest"],
            batch_id="batch-rerun",
            batch_index=1,
            event_seqs=[1, 2],
        )
        rerun_payload = _payload(spec_bundle["spec"], "session-test", installed["snapshot_digest"])
        rerun_payload["batch_id"] = "batch-rerun"
        ledger.commit_batch(rerun_payload)
        after_rerun = ledger.recover_session("session-test")
        assert after_rerun.unknown_batch_ids == ()
        assert after_rerun.committed_batch_indices == (1,)
        assert after_rerun.resume_after_seq == 2
    finally:
        ledger.close()


def test_replay_mismatch_locates_first_event_db_and_field(tmp_path, spec_bundle):
    ledger, installed, payload, committed = _committed_ledger(tmp_path, spec_bundle["spec"])
    try:
        recomputed = copy.deepcopy(payload)
        recomputed["events"][0]["route_record"]["new_shard"] = "orders:s3"
        with pytest.raises(DiagnosticError) as caught:
            ledger.replay_against("session-test", [recomputed])
        error = caught.value
        assert error.code == ErrorCode.REPLAY.value
        assert error.details["event_seq"] == 1
        assert error.details["db_id"] == "db-a"
        assert error.details["field"] == "new_shard"
        assert error.details["batch_index"] == 1
    finally:
        ledger.close()


def test_ledger_content_digest_detects_tampering(tmp_path, spec_bundle):
    ledger, installed, payload, committed = _committed_ledger(tmp_path, spec_bundle["spec"])
    try:
        tampered = copy.deepcopy(payload)
        tampered["events"][0]["hit_shards"] = ["orders:s3"]
        ledger.connection.execute(
            "UPDATE batches SET payload_json=? WHERE session_id='session-test' AND batch_index=1",
            (json.dumps(tampered, sort_keys=True, separators=(",", ":")),),
        )
        with pytest.raises(DiagnosticError) as caught:
            ledger.verify_integrity("session-test")
        assert caught.value.code == ErrorCode.REPLAY.value
        assert caught.value.field == "content_digest"
    finally:
        ledger.close()

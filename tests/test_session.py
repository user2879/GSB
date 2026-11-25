from __future__ import annotations

import copy

import pytest

from shard_diag.errors import DiagnosticError, ErrorCode
from shard_diag.ledger import Ledger
from shard_diag.routing import generate_sample_keys, sample_hit_shards
from shard_diag.session import DiagnosticSession


def _run_layout(spec_bundle, ledger_path, *, batch_size, startup_order, session_id):
    session = DiagnosticSession(
        spec_bundle["spec"],
        spec_bundle["current_path"],
        ledger_path,
        session_id=session_id,
    )
    try:
        session.start_databases(startup_order)
        report = session.run_batches(batch_size=batch_size, sample_count=2)
        replay = session.replay_ledger()
        return report.routing_bytes, report.route_records, replay
    finally:
        session.close()


def test_batch_split_and_startup_order_do_not_change_route_bytes(tmp_path, spec_bundle):
    first_ledger = tmp_path / "layout-a.sqlite"
    second_ledger = tmp_path / "layout-b.sqlite"
    bytes_a, records_a, replay_a = _run_layout(
        spec_bundle,
        first_ledger,
        batch_size=1,
        startup_order=["db-b", "db-a"],
        session_id="layout-a",
    )
    bytes_b, records_b, replay_b = _run_layout(
        spec_bundle,
        second_ledger,
        batch_size=4,
        startup_order=["db-a", "db-b"],
        session_id="layout-b",
    )
    assert bytes_a == bytes_b
    assert records_a == records_b
    assert replay_a["verified"] and replay_b["verified"]
    assert [record["event_seq"] for record in records_a] == [1, 2, 3, 4, 5, 6]


def test_forced_kill_recovery_unknown_batch_and_snapshot_replacement(tmp_path, spec_bundle):
    ledger_path = tmp_path / "session.sqlite"
    Ledger.reset(ledger_path)
    session = DiagnosticSession(
        spec_bundle["spec"],
        spec_bundle["current_path"],
        ledger_path,
        session_id="session-kill",
    )
    try:
        session.start_databases(["db-b", "db-a"])
        first = session.run_batches(
            list(spec_bundle["spec"].events[:3]),
            batch_size=2,
            sample_count=2,
        )
        assert len(first.batches) == 2
        next_index = session.ledger.get_progress(session.session_id)["last_batch_index"] + 1
        remaining = list(spec_bundle["spec"].events[3:])
        crash_intent = session.ledger.begin_batch(
            session_id=session.session_id,
            snapshot_digest=session.snapshot_digest,
            batch_id="batch-crash-intent",
            batch_index=next_index,
            event_seqs=[event.seq for event in remaining],
        )
        crash_event = remaining[0]
        rule = spec_bundle["spec"].event_rule(crash_event)
        db = spec_bundle["spec"].database(crash_event.database)
        keys = generate_sample_keys(
            seed=db.seed,
            event_seq=crash_event.seq,
            route_key=crash_event.route_key,
            rule=rule,
            sample_count=2,
        )
        shards = sample_hit_shards(rule, keys)
        from shard_diag.leases import LeaseRequest

        for shard in shards:
            session.lease_manager.acquire(
                LeaseRequest(crash_event.database, shard, crash_event.seq, db.route_version),
                ttl_ms=10_000,
            )
        assert session.lease_manager.snapshot()["active_leases"] == len(shards)
        session.force_kill_database(crash_event.database)
        recovery = session.recover_after_worker_restart([crash_event.database])
        assert recovery.unknown_batch_ids == ("batch-crash-intent",)
        assert recovery.unknown_event_seqs == tuple(event.seq for event in remaining)
        assert recovery.exit_report_available is False
        assert recovery.resume_after_seq == 3
        assert session.lease_manager.snapshot()["active_leases"] == 0

        resumed = session.run_batches(batch_size=2, sample_count=2, resume=True)
        assert len(resumed.batches) == 2
        replay = session.replay_ledger()
        assert replay["verified"]
        assert session.ledger.recover_session(session.session_id).unknown_batch_ids == ()
        assert len(session.ledger.read_batches(session.session_id)) == 4

        generation_before = session.snapshot_generation
        old_batches = len(session.ledger.read_batches(session.session_id))
        session.replace_snapshot(
            spec_bundle["edited_spec"],
            spec_path=spec_bundle["edited_path"],
            generation=generation_before + 1,
        )
        assert session.snapshot_generation == generation_before + 1
        assert len(session.ledger.read_batches(session.session_id)) == 0
        assert old_batches == 4
        session.start_databases(["db-a", "db-b"])
        edited = session.run_batches(batch_size=1, sample_count=1)
        assert len(edited.batches) == len(spec_bundle["edited_spec"].events)
        assert session.replay_ledger()["verified"]
    finally:
        session.close()


def test_invalid_subset_is_reprobed_through_evolved_worker_spec(tmp_path, spec_bundle):
    ledger_path = tmp_path / "evolve.sqlite"
    session = DiagnosticSession(
        spec_bundle["spec"],
        spec_bundle["current_path"],
        ledger_path,
        session_id="session-evolve",
    )
    try:
        session.start_databases()
        session.run_batches(batch_size=3, sample_count=2)
        report = session.invalidate(
            spec_bundle["spec"].rule("orders", 2),
            spec_bundle["evolved_spec"].rule("orders", 3),
        )
        assert report.invalid_event_seqs
        result = session.reprobe_invalid_subset(
            report,
            spec_bundle["evolved_spec"],
            spec_bundle["evolved_path"],
            sample_count=2,
        )
        assert result.probed_event_seqs == report.invalid_event_seqs
        assert result.recompute.unaffected_byte_identical
        assert result.unaffected_responses_byte_identical
        assert result.recompute.unaffected_before_digest == result.recompute.unaffected_after_digest
        recomputed = {
            (record["event_seq"], record["db_id"], record["table"], record["route_key"])
            for record in result.route_records
            if record["new_route_version"] == 3
        }
        assert {key[0] for key in recomputed} == set(report.invalid_event_seqs)
    finally:
        session.close()


def test_session_replay_reports_first_mismatched_field(tmp_path, spec_bundle):
    ledger_path = tmp_path / "mismatch.sqlite"
    session = DiagnosticSession(
        spec_bundle["spec"],
        spec_bundle["current_path"],
        ledger_path,
        session_id="session-mismatch",
    )
    try:
        session.start_databases()
        session.run_batches(batch_size=3, sample_count=1)
        stored = session.ledger.read_batches(session.session_id)[0]["payload"]
        tampered = copy.deepcopy(stored)
        tampered["events"][0]["route_record"]["new_shard"] = "orders:s9"
        session.ledger.replace_batch_for_test(session.session_id, 1, tampered)
        with pytest.raises(DiagnosticError) as caught:
            session.replay_ledger()
        error = caught.value
        assert error.code == ErrorCode.REPLAY.value
        assert error.details["event_seq"] == 1
        assert error.details["db_id"] == "db-a"
        assert error.details["field"] == "new_shard"
    finally:
        session.close()

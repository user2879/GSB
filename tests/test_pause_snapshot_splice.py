from __future__ import annotations

import dataclasses
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from probelease import GrantError, LeaseOrchestrator, compare_splice_ledgers
from probelease.declarations import declaration_from_dicts, load_declarations
from probelease.ledger import Ledger, restore_from_snapshot

from conftest import SAMPLES, wait_ports_free


def test_inflight_pause_marks_dangling_and_ignores_them_on_resume(orchestrator):
    result = orchestrator.execute_round("alpha", pause_after_send=True)
    assert result.dangling is True
    assert result.committed is False
    assert orchestrator.query_lease("alpha")["cycle_cursor"] == 0
    assert orchestrator.ledger.round_records("alpha") == []
    raw_rows = [
        dict(row)
        for row in orchestrator.ledger._conn.execute(
            """
            SELECT cycle, attempt, probe_id, outcome, reason, send_mono_ns, round_id
            FROM datagrams WHERE session_id = 'alpha' ORDER BY probe_id
            """
        )
    ]
    assert len(raw_rows) == 3
    assert {row["outcome"] for row in raw_rows} == {"DANGLING_UNKNOWN"}
    assert all(row["send_mono_ns"] is not None for row in raw_rows)
    assert all(row["round_id"] is None for row in raw_rows)
    event = orchestrator.ledger.pause_events("alpha")[0]
    assert event["reason"] == "USER_PAUSE_INFLIGHT"
    assert len(event["details"]["dangling_tx_ids"]) == 3

    orchestrator.resume_session("alpha")
    resumed = orchestrator.execute_round("alpha")
    assert (resumed.cycle, resumed.attempt, resumed.ratio_bp) == (1, 2, 6667)
    assert [record["cycle"] for record in orchestrator.ledger.round_records("alpha")] == [1]


def test_byte_identical_splice_across_process_restart(declarations, tmp_path):
    baseline_path = tmp_path / "baseline.sqlite3"
    baseline = LeaseOrchestrator(declarations, baseline_path)
    baseline.start_probes()
    baseline.run_cycles("alpha", 4)
    baseline.close()
    wait_ports_free(declarations)

    spliced_path = tmp_path / "spliced.sqlite3"
    first = LeaseOrchestrator(declarations, spliced_path)
    first.start_probes()
    first.run_cycles("alpha", 2)
    first.pause_session("alpha", "SPLICE_PAUSE")
    first.close()
    wait_ports_free(declarations)

    second = LeaseOrchestrator(declarations, spliced_path)
    second.start_probes()
    second.resume_session("alpha")
    second.run_cycles("alpha", 2)
    second.close()
    wait_ports_free(declarations)

    with Ledger(baseline_path, declarations, recover_incomplete=False) as left:
        with Ledger(spliced_path, declarations, recover_incomplete=False) as right:
            report = compare_splice_ledgers(left, right, session_ids=["alpha"])
            assert report.equal is True
            assert report.left_sha256 == report.right_sha256
            assert report.bytes_compared > 1000
            reasons = [
                (event["kind"], event["reason"], event["cycle"])
                for event in right.pause_events("alpha")
            ]
            assert reasons == [
                ("PAUSE", "SPLICE_PAUSE", 3),
                ("RESUME", "USER_RESUME", 3),
            ]


def test_snapshot_wholesale_replace_and_readonly_backup(declarations, tmp_path):
    ledger_path = tmp_path / "ledger.sqlite3"
    snapshot_path = tmp_path / "snapshot.json"
    orchestrator = LeaseOrchestrator(declarations, ledger_path)
    orchestrator.start_probes()
    orchestrator.execute_round("alpha")
    saved = orchestrator.save_snapshot(snapshot_path)
    orchestrator.execute_round("alpha")
    orchestrator.close()
    wait_ports_free(declarations)

    assert saved["state_digest"]
    report = restore_from_snapshot(
        snapshot_path=snapshot_path,
        db_path=ledger_path,
        declarations=declarations,
    )
    assert report.replaced is True
    assert report.backup_path is not None
    assert os.stat(report.backup_path).st_mode & 0o222 == 0
    with Ledger(ledger_path, declarations, recover_incomplete=False) as ledger:
        assert [record["cycle"] for record in ledger.round_records("alpha")] == [1]
        assert ledger.get_session("alpha")["lease_version"] == 2

    noop = restore_from_snapshot(
        snapshot_path=snapshot_path,
        db_path=ledger_path,
        declarations=declarations,
    )
    assert noop.replaced is False


def test_restore_lists_added_sessions_and_probes(declarations, tmp_path):
    full_path = tmp_path / "full.sqlite3"
    full = LeaseOrchestrator(declarations, full_path)
    full.start_probes()
    full.execute_round("alpha")
    snapshot_path = tmp_path / "full.json"
    full.save_snapshot(snapshot_path)
    full.close()
    wait_ports_free(declarations)

    sessions_doc = json.loads((SAMPLES / "sessions.json").read_text(encoding="utf-8"))
    probes_doc = json.loads((SAMPLES / "probes.json").read_text(encoding="utf-8"))
    policy_doc = json.loads((SAMPLES / "ack_policy.json").read_text(encoding="utf-8"))
    sessions_doc["session_set_version"] = 99
    sessions_doc["sessions"] = [sessions_doc["sessions"][0]]
    probes_doc["probe_parameter_version"] = 99
    probes_doc["probes"] = {
        "alpha_ok_1": probes_doc["probes"]["alpha_ok_1"],
        "alpha_ok_2": probes_doc["probes"]["alpha_ok_2"],
        "alpha_drop": probes_doc["probes"]["alpha_drop"],
    }
    subset = declaration_from_dicts(sessions_doc, probes_doc, policy_doc, tmp_path / "subset")
    subset_path = tmp_path / "subset.sqlite3"
    Ledger(subset_path, subset).close()

    report = restore_from_snapshot(
        snapshot_path=snapshot_path,
        db_path=subset_path,
        declarations=declarations,
    )
    assert report.replaced is True
    assert set(report.added_sessions) == {"lossy", "late", "mismatch", "child"}
    assert set(report.added_probes) == {
        "lossy_ok",
        "lossy_drop",
        "late_ok_but_late",
        "bad_sequence",
        "child_ok",
    }


def test_snapshot_descriptor_mismatch_is_grant_error(declarations, tmp_path):
    orchestrator = LeaseOrchestrator(declarations, tmp_path / "ledger.sqlite3")
    orchestrator.start_probes()
    orchestrator.execute_round("alpha")
    snapshot_path = tmp_path / "snapshot.json"
    orchestrator.save_snapshot(snapshot_path)
    orchestrator.close()
    wait_ports_free(declarations)

    sessions_doc = json.loads((SAMPLES / "sessions.json").read_text(encoding="utf-8"))
    probes_doc = json.loads((SAMPLES / "probes.json").read_text(encoding="utf-8"))
    policy_doc = json.loads((SAMPLES / "ack_policy.json").read_text(encoding="utf-8"))
    sessions_doc["session_set_version"] = 4
    changed = declaration_from_dicts(sessions_doc, probes_doc, policy_doc, tmp_path / "changed")
    with pytest.raises(GrantError) as caught:
        restore_from_snapshot(
            snapshot_path=snapshot_path,
            db_path=tmp_path / "ledger.sqlite3",
            declarations=changed,
        )
    assert caught.value.code == "E_GRANT"


def test_snapshot_restore_discards_uncommitted_staging(declarations, tmp_path):
    ledger_path = tmp_path / "ledger.sqlite3"
    snapshot_path = tmp_path / "snapshot.json"
    orchestrator = LeaseOrchestrator(declarations, ledger_path)
    orchestrator.start_probes()
    orchestrator.execute_round("alpha")
    orchestrator.save_snapshot(snapshot_path)
    orchestrator.close()
    wait_ports_free(declarations)

    staged = Ledger(ledger_path, declarations, recover_incomplete=False)
    try:
        now = time.monotonic_ns()
        staged.stage_attempt(
            session_id="alpha",
            cycle=2,
            attempt=1,
            records=[
                {
                    "probe_id": "alpha_ok_1",
                    "tx_id": "staged-only-tx",
                    "request_payload": "{}",
                    "request_lease_version": 2,
                    "planned_mono_ns": now,
                }
            ],
            started_mono_ns=now,
        )
    finally:
        staged.close()

    report = restore_from_snapshot(
        snapshot_path=snapshot_path,
        db_path=ledger_path,
        declarations=declarations,
    )
    assert report.replaced is True
    assert any(item["field"] == "ledger.staging" for item in report.descriptor_mismatches)
    with sqlite3.connect(str(ledger_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM _stage_attempts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM _stage_datagrams").fetchone()[0] == 0

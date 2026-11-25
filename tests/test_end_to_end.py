from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time

from lease_arbiter.canonical import canonical_bytes, digest_text
from lease_arbiter.recovery import replay_against_ledger
from lease_arbiter.runtime import run_live
from lease_arbiter.verifier import verify_spec

from .conftest import ROOT, write_local_spec


def test_live_paths_match_independent_verifier_and_replay(tmp_path):
    spec = write_local_spec(tmp_path)
    verified = verify_spec(spec)
    expected_bytes = canonical_bytes(verified["comparison"]["sequential_projection"])

    sequential = run_live(
        spec,
        ledger_path=tmp_path / "sequential.sqlite3",
        requested_path="sequential",
        fresh=True,
    )
    batch = run_live(
        spec,
        ledger_path=tmp_path / "batch.sqlite3",
        requested_path="batch",
        fresh=True,
    )
    sequential_bytes = canonical_bytes(sequential["comparison_projection"])
    batch_bytes = canonical_bytes(batch["comparison_projection"])
    assert sequential_bytes == expected_bytes
    assert batch_bytes == expected_bytes
    assert sequential_bytes == batch_bytes
    assert sequential["runtime"]["ledger"]["snapshot_version"] == len(spec.events) + 1
    assert batch["runtime"]["ledger"]["snapshot_version"] == len(spec.events) + 1

    replay = replay_against_ledger(spec, tmp_path / "sequential.sqlite3")
    assert replay["status"] == "replay_ok"
    assert replay["committed_events"] == len(spec.events)
    assert replay["unknown_request_count"] == 0

    with sqlite3.connect(tmp_path / "sequential.sqlite3") as conn:
        commits = conn.execute(
            "SELECT format_version, declaration_version, payload_json, content_digest FROM commits ORDER BY event_seq"
        ).fetchall()
        ownership = conn.execute(
            "SELECT event_seq, lease_id, format_version, payload_json, content_digest FROM ownership ORDER BY ownership_id"
        ).fetchall()
    assert len(commits) == len(spec.events)
    assert len(ownership) == 8
    assert all(row[0] == 1 and row[1] == spec.declaration_version for row in commits)
    assert all(digest_text(row[2]) == row[3] for row in commits)
    assert all(row[2] == 1 and digest_text(row[3]) == row[4] for row in ownership)
    assert {row[0] for row in ownership} == {1, 2, 3, 6, 7, 9, 10}


def test_forced_termination_leaves_pending_unknown_and_replay_uses_committed_prefix(tmp_path):
    spec = write_local_spec(tmp_path)
    ledger_path = tmp_path / "crash.sqlite3"
    command = [
        sys.executable,
        "-m",
        "lease_arbiter",
        "run",
        "--nodes",
        spec.nodes_file,
        "--leases",
        spec.leases_file,
        "--events",
        spec.events_file,
        "--ledger",
        str(ledger_path),
        "--path",
        "sequential",
        "--fresh",
        "--crash-after-event",
        "5",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    exit_code = process.wait(timeout=15)
    if exit_code != 137:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise AssertionError(f"crash child exited with {exit_code}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(0.25)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(0.1)

    replay = replay_against_ledger(spec, ledger_path)
    assert replay["status"] == "replay_ok"
    assert replay["committed_events"] == 4
    assert replay["unknown_request_count"] == 1
    assert replay["unknown_requests"][0]["event_seq"] == 5
    assert replay["snapshot_version"] == 5

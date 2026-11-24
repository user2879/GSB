from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from probelease import LeaseOrchestrator, SpliceError, replay_verify
from probelease.errors import ProbeLeaseError
from probelease.ledger import Ledger, restore_from_snapshot

from conftest import ROOT, SAMPLES, wait_ports_free


def _cli(*args: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "probelease.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_forced_termination_snapshot_restore_and_replay(declarations, tmp_path):
    ledger_path = tmp_path / "crash.sqlite3"
    snapshot_path = tmp_path / "before.json"
    log_path = tmp_path / "cli.jsonl"
    crashed = _cli(
        "run",
        "--sessions",
        str(SAMPLES / "sessions.json"),
        "--probes",
        str(SAMPLES / "probes.json"),
        "--ack-policy",
        str(SAMPLES / "ack_policy.json"),
        "--ledger",
        str(ledger_path),
        "--session",
        "alpha",
        "--cycles",
        "2",
        "--snapshot",
        str(snapshot_path),
        "--snapshot-after-cycle",
        "1",
        "--crash-before-commit-cycle",
        "2",
        "--log",
        str(log_path),
    )
    assert crashed.returncode == 137
    assert crashed.stdout == ""
    wait_ports_free(declarations)
    conn = sqlite3.connect(str(ledger_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM _stage_attempts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM datagrams").fetchone()[0] == 3
    finally:
        conn.close()

    recovered = LeaseOrchestrator(declarations, ledger_path)
    try:
        report = recovered.recovery_report.to_dict()
        assert report["rerun_required"] == [
            {
                "session_id": "alpha",
                "cycle": 2,
                "attempt": 1,
                "action": "mark_unknown_rerun_required",
            }
        ]
        unknown = [
            dict(row)
            for row in recovered.ledger._conn.execute(
                "SELECT probe_id, outcome FROM datagrams WHERE outcome = 'UNKNOWN' ORDER BY probe_id"
            )
        ]
        assert len(unknown) == 3
        assert recovered.ledger._conn.execute(
            "SELECT COUNT(*) FROM _stage_datagrams"
        ).fetchone()[0] == 0
    finally:
        recovered.close()

    restored = restore_from_snapshot(
        snapshot_path=snapshot_path,
        db_path=ledger_path,
        declarations=declarations,
    )
    assert restored.replaced is True
    wait_ports_free(declarations)
    continuation = LeaseOrchestrator(declarations, ledger_path)
    continuation.start_probes()
    try:
        assert continuation.execute_round("alpha").cycle == 2
        assert continuation.execute_round("alpha").cycle == 3
    finally:
        continuation.close()
    wait_ports_free(declarations)

    replay_report = replay_verify(ledger_path, declarations, session_id="alpha")
    assert replay_report.rounds_verified == 3
    assert replay_report.probes_verified == 9
    wait_ports_free(declarations)

    bad_path = tmp_path / "bad_replay.sqlite3"
    shutil.copy2(ledger_path, bad_path)
    _tamper_ratio(bad_path, cycle=2, value=1)
    with pytest.raises(ProbeLeaseError) as caught:
        replay_verify(bad_path, declarations, session_id="alpha")
    assert caught.value.code == "E_REPLAY"
    assert caught.value.details["field"] == "ratio_bp"
    assert caught.value.details["cycle"] == 2


def test_cli_fault_injections_log_monotonic_time(tmp_path):
    expected = {
        "corrupt-session": "E_SESSION",
        "ack-version": "E_GRANT",
        "dependency-cycle": "E_DEPEND",
    }
    for injection, code in expected.items():
        log_path = tmp_path / f"{injection}.jsonl"
        completed = _cli(
            "validate",
            "--sessions",
            str(SAMPLES / "sessions.json"),
            "--probes",
            str(SAMPLES / "probes.json"),
            "--ack-policy",
            str(SAMPLES / "ack_policy.json"),
            "--inject",
            injection,
            "--log",
            str(log_path),
        )
        assert completed.returncode == 2
        payload = json.loads(completed.stdout)
        assert payload["code"] == code
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert records[-1]["code"] == code
        assert isinstance(records[-1]["monotonic_ns"], int)


def test_splice_error_locates_first_difference(declarations, tmp_path):
    paths = []
    for name in ("left", "right"):
        path = tmp_path / f"{name}.sqlite3"
        instance = LeaseOrchestrator(declarations, path)
        instance.start_probes()
        instance.run_cycles("alpha", 2)
        instance.close()
        wait_ports_free(declarations)
        paths.append(path)
    with Ledger(paths[0], declarations, recover_incomplete=False) as left:
        with Ledger(paths[1], declarations, recover_incomplete=False) as right:
            from probelease import compare_splice_ledgers

            assert compare_splice_ledgers(left, right, session_ids=["alpha"]).equal
    _tamper_ratio(paths[1], cycle=2, value=1)
    with Ledger(paths[0], declarations, recover_incomplete=False) as left:
        with Ledger(paths[1], declarations, recover_incomplete=False) as right:
            with pytest.raises(SpliceError) as caught:
                compare_splice_ledgers(left, right, session_ids=["alpha"])
    assert caught.value.code == "E_SPLICE"
    assert caught.value.details["cycle"] == 2
    assert caught.value.details["field"] == "ratio_bp"


def _tamper_ratio(path: Path, *, cycle: int, value: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "UPDATE rounds SET ratio_bp = ? WHERE session_id = 'alpha' AND cycle = ?",
            (value, cycle),
        )
        conn.commit()
    finally:
        conn.close()


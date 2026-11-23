from __future__ import annotations

import json
import os
import subprocess
import sys

from .conftest import ROOT, SAMPLES


def cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "leasebench", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


def test_cli_error_codes_and_structured_logs(tmp_path) -> None:
    log = tmp_path / "errors.log"
    cycle_db = tmp_path / "cycle.sqlite"
    cycle = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(cycle_db),
        "--inject-cycle",
        "--log",
        str(log),
    )
    assert cycle.returncode == 2
    assert '"code": "E_DECL"' in cycle.stdout

    quota_db = tmp_path / "quota.sqlite"
    quota = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(quota_db),
        "--inject-quota",
        "--log",
        str(log),
    )
    assert quota.returncode == 3
    assert '"code": "E_QUOTA"' in quota.stdout

    snap_db = tmp_path / "snap.sqlite"
    first = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(snap_db),
        "--limit-events",
        "6",
    )
    assert first.returncode == 0
    snapshot = cli(
        "replay",
        "--db",
        str(snap_db),
        "--inject-snapshot",
        "--log",
        str(log),
    )
    assert snapshot.returncode == 4
    assert '"code": "E_SNAPSHOT"' in snapshot.stdout

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    codes = {record.get("code") for record in records}
    assert {"E_DECL", "E_QUOTA", "E_SNAPSHOT"} <= codes


def test_cli_forced_termination_and_recovery(tmp_path) -> None:
    db = tmp_path / "crash.sqlite"
    log = tmp_path / "crash.log"
    crashed = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(db),
        "--inject-crash-after-pending",
        "--log",
        str(log),
    )
    assert crashed.returncode == 17

    pending_text = tmp_path / "pending.txt"
    pending_view = cli(
        "view", "--db", str(db), "--text-out", str(pending_text)
    )
    assert pending_view.returncode == 0
    assert "UNKNOWN_RECOMPUTE_REQUIRED" in pending_text.read_text(encoding="utf-8")

    out = tmp_path / "recovered.json"
    recovered = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(db),
        "--json-out",
        str(out),
        "--log",
        str(log),
    )
    assert recovered.returncode == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["recovery_status"] == "CLEAN"
    assert result["unknown_lease_ids"] == []
    assert result["recomputed_unknown_lease_ids"]
    assert result["final_cursor"] == 11


def test_cli_headless_view_is_repeatable(tmp_path) -> None:
    db = tmp_path / "view.sqlite"
    first = cli(
        "run",
        "--declaration",
        str(SAMPLES / "declarations_v1.json"),
        "--events",
        str(SAMPLES / "events.ndjson"),
        "--db",
        str(db),
    )
    assert first.returncode == 0
    one = cli("ui", "--db", str(db), "--headless")
    two = cli("ui", "--db", str(db), "--headless")
    assert one.returncode == 0
    assert one.stdout == two.stdout
    assert "SERVERLESS CONCURRENCY LEASE WORKBENCH" in one.stdout

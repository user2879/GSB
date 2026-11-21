#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "demo"
PYTHON = sys.executable


def invoke(arguments: list[str], *, expected: int = 0) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    completed = subprocess.run(
        [PYTHON, "-m", "mvcc_workbench.cli", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    if completed.returncode != expected:
        raise RuntimeError(
            f"command failed: {arguments}\n"
            f"expected={expected} actual={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed, payload


def clean_ledger(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)
    ledger = DEMO / "ledger.sqlite"
    event_log = DEMO / "events.jsonl"
    clean_ledger(ledger)
    if event_log.exists():
        event_log.unlink()

    base = str(ROOT / "samples" / "base.json")
    evolved = str(ROOT / "samples" / "evolved.json")
    common = ["--db", str(ledger), "--log", str(event_log), "--json"]

    _, session1 = invoke(["run", "--input", base, *common])
    assert session1 is not None
    assert session1["status"] == "COLD_START_COMPLETE"
    assert session1["code"] == "OK"
    assert session1["generation"] == 1
    assert len(session1["rounds"]) == 3
    assert len(session1["decisions"]) == 8
    write_json(DEMO / "session1_cold.json", session1)

    _, session2 = invoke(["run", "--input", evolved, *common])
    assert session2 is not None
    assert session2["status"] == "EVOLUTION_REPLACED"
    assert session2["code"] == "E_REPLAY"
    assert session2["generation"] == 2
    assert session2["affected"]["affected_query_ids"] == ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"]
    unaffected = {item["query_id"]: item for item in session2["affected"]["unchanged_checks"]}
    assert unaffected["Q07"]["byte_equal"] is True
    assert unaffected["Q08"]["byte_equal"] is True
    write_json(DEMO / "session2_evolution.json", session2)

    _, session3 = invoke(["run", "--input", evolved, *common])
    assert session3 is not None
    assert session3["status"] == "REPLAY_OK"
    assert session3["code"] == "OK"
    assert session3["generation"] == 2
    assert session3["ledger_version"]["ledger_cursor"] == session2["ledger_version"]["ledger_cursor"]
    write_json(DEMO / "session3_replay.json", session3)

    crash_ledger = DEMO / "crash-ledger.sqlite"
    clean_ledger(crash_ledger)
    _, crashed = invoke(
        [
            "run",
            "--input",
            base,
            "--db",
            str(crash_ledger),
            "--log",
            str(event_log),
            "--json",
            "--crash-in-round",
            "2",
            "--crash-after-submit",
            "1",
        ],
        expected=87,
    )
    assert crashed is None
    _, recovery = invoke(
        ["recover", "--input", base, "--db", str(crash_ledger), "--log", str(event_log), "--json"]
    )
    assert recovery is not None
    assert recovery["status"] == "RECOVERY_REQUIRED"
    assert recovery["report_available"] is False
    assert recovery["next_action"] == "RERUN_UNKNOWN_TASKS"
    _, resumed = invoke(
        ["run", "--input", base, "--db", str(crash_ledger), "--log", str(event_log), "--json"]
    )
    assert resumed is not None
    assert resumed["status"] == "RESUMED_COMPLETE"
    assert resumed["generation"] == 1
    assert len(resumed["decisions"]) == 8
    assert resumed["recovery"]["unknown_count"] == 0
    write_json(DEMO / "crash_recovery.json", {"recovery": recovery, "resumed": resumed})

    injections = {}
    cases = (
        ("chain_cycle", "invalid_chain_cycle.json", "E_CHAIN", 3),
        ("txn_commit_regress", "invalid_txn_commit_regression.json", "E_TXN", 2),
        ("snapshot_digest", "invalid_snapshot_digest.json", "E_SNAPSHOT", 6),
    )
    for kind, filename, code, exit_code in cases:
        inject_db = DEMO / f"inject-{kind}.sqlite"
        clean_ledger(inject_db)
        _, payload = invoke(
            [
                "inject",
                "--kind",
                kind,
                "--input",
                str(ROOT / "samples" / filename),
                "--db",
                str(inject_db),
                "--log",
                str(event_log),
                "--json",
            ],
            expected=exit_code,
        )
        assert payload is not None
        assert payload["code"] == code
        injections[kind] = {"expected_code": code, "exit_code": exit_code, "location": payload["location"]}

    summary = {
        "ok": True,
        "python": sys.version.split()[0],
        "ui_mode": "CLI_HEADLESS" if not _tkinter_available() else "TK_AVAILABLE",
        "sessions": {
            "cold": {
                "status": session1["status"],
                "generation": session1["generation"],
                "ledger_cursor": session1["ledger_version"]["ledger_cursor"],
                "conclusion_digest": session1["conclusion_digest"],
            },
            "evolution": {
                "status": session2["status"],
                "code": session2["code"],
                "generation": session2["generation"],
                "affected_query_ids": session2["affected"]["affected_query_ids"],
                "unaffected_byte_equal": all(item["byte_equal"] for item in session2["affected"]["unchanged_checks"]),
                "first_mismatch": session2["diagnostic"]["location"],
            },
            "replay": {
                "status": session3["status"],
                "generation": session3["generation"],
                "ledger_cursor": session3["ledger_version"]["ledger_cursor"],
            },
        },
        "crash_recovery": {
            "crash_exit": 87,
            "recovery_status": recovery["status"],
            "report_available_before_rerun": recovery["report_available"],
            "resume_status": resumed["status"],
            "unknown_after_resume": resumed["recovery"]["unknown_count"],
        },
        "injections": injections,
        "event_log": str(event_log),
    }
    write_json(DEMO / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())

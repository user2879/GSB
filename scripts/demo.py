#!/usr/bin/env python3
"""Run the complete deterministic experiment sequence in the current directory."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "samples/session.json"
EVOLVED_CONFIG = "samples/session_evolved.json"
RECORDS = "samples/records.jsonl"
EXPECTED = "samples/expected_summary.json"
LEDGER = "state/demo-ledger.sqlite3"
CRASH_LEDGER = "state/demo-crash.sqlite3"
REPLAY_LEDGER = "state/demo-replay.sqlite3"
SNAPSHOT_DIR = "snapshots/demo"
BACKUP_DIR = "backup"
OUT = "out-demo"
EVOLVED_OUT = "out-demo-evolved"
LOG = "logs/demo-structured.jsonl"


def command(*args: str, expected: int = 0) -> dict[str, Any]:
    full = [sys.executable, "-m", "mwlab", *args, "--log-file", LOG]
    process = subprocess.run(full, cwd=ROOT, text=True, capture_output=True, check=False)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] = {}
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = {"raw_stdout": process.stdout}
    if process.returncode != expected:
        raise RuntimeError(
            f"command failed: {' '.join(full)}\n"
            f"expected={expected} actual={process.returncode}\n"
            f"stdout={process.stdout}\nstderr={process.stderr}"
        )
    return payload


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    print("[1/9] selfcheck")
    selfcheck = command("selfcheck", "--config", CONFIG)
    check(selfcheck["status"] in {"pass", "pass_with_warnings"}, "selfcheck failed")

    print("[2/9] initialize writable paths")
    command("init", "--config", CONFIG)

    print("[3/9] normalize, derive semantics, commit topic batches")
    base = command(
        "run",
        "--config",
        CONFIG,
        "--records",
        RECORDS,
        "--ledger",
        LEDGER,
        "--out",
        OUT,
        "--expected",
        EXPECTED,
    )
    check(base["verdict_counts"] == {"at_least_once": 8, "at_most_once": 1, "exactly_once": 1}, "verdict count mismatch")
    check(base["conflict_counts"] == {
        "ACK_WITH_UNCOVERED_RANGE": 2,
        "DUP_IDEMPOTENCY_ACROSS_GROUPS": 1,
        "SEMANTIC_MARKER_MISMATCH": 1,
    }, "conflict count mismatch")
    check(base["stats_digest"] == "eff2d0b04ba5e71b26da5ab4fa868ac22863ed29dd96d7c5df5f0c63eb59a175", "stats digest mismatch")

    print("[4/9] fixed-point report and independent replay")
    command("report", "--artifact", f"{OUT}/artifact.json", "--out", f"{OUT}/report.json")
    verify = command(
        "verify",
        "--config",
        CONFIG,
        "--records",
        RECORDS,
        "--model-digest",
        f"{OUT}/model_digest.txt",
        "--conclusions",
        f"{OUT}/conclusions.jsonl",
        "--group-conclusions",
        f"{OUT}/group_conclusions.jsonl",
        "--conflicts",
        f"{OUT}/conflicts.jsonl",
        "--stats",
        f"{OUT}/stats.json",
    )
    check(verify["byte_identical"], "independent replay is not byte-identical")

    print("[5/9] snapshot save/restore")
    saved = command(
        "snapshot",
        "save",
        "--config",
        CONFIG,
        "--ledger",
        LEDGER,
        "--dir",
        SNAPSHOT_DIR,
        "--backup-dir",
        BACKUP_DIR,
        "--name",
        "baseline",
    )
    restored = command(
        "snapshot",
        "restore",
        "--config",
        CONFIG,
        "--ledger",
        LEDGER,
        "--dir",
        SNAPSHOT_DIR,
        "--backup-dir",
        BACKUP_DIR,
        "--name",
        "baseline",
    )
    check(restored["replacement_mode"] == "wholesale_copy_no_merge", "restore was not wholesale")
    check(saved["content_digest"] == restored["content_digest"], "snapshot digest changed on restore")

    print("[6/9] complete-ledger recovery")
    recovered = command(
        "recover",
        "--ledger",
        LEDGER,
        "--records",
        RECORDS,
    )
    check(recovered["status"] == "complete" and recovered["replay_ok"], "complete recovery failed")

    print("[7/9] forced termination, unknown messages, and rerun")
    command(
        "run",
        "--config",
        CONFIG,
        "--records",
        RECORDS,
        "--ledger",
        CRASH_LEDGER,
        "--out",
        "out-demo-crash",
        "--crash-after-topic",
        "1",
        expected=91,
    )
    incomplete = command("recover", "--ledger", CRASH_LEDGER, "--records", RECORDS)
    check(incomplete["status"] == "incomplete", "crash recovery should be incomplete")
    check(not incomplete["exit_report_available"], "incomplete recovery must not claim an exit report")
    check(incomplete["unknown_messages"], "uncommitted messages must be unknown")
    command(
        "run",
        "--config",
        CONFIG,
        "--records",
        RECORDS,
        "--ledger",
        CRASH_LEDGER,
        "--out",
        "out-demo-crash",
    )
    complete = command("recover", "--ledger", CRASH_LEDGER, "--records", RECORDS)
    check(complete["status"] == "complete" and complete["replay_ok"], "rerun recovery failed")

    print("[8/9] negative-path injections: E_FORMAT, E_DUPKEY, E_SNAPSHOT, E_REPLAY, E_DIVERGE")
    format_error = command(
        "run", "--config", CONFIG, "--records", RECORDS, "--inject-format", expected=3
    )["error"]
    check(format_error["code"] == "E_FORMAT", "format injection code mismatch")
    dupkey_error = command(
        "run", "--config", CONFIG, "--records", RECORDS, "--inject-dupkey", expected=4
    )["error"]
    check(dupkey_error["code"] == "E_DUPKEY", "duplicate-key injection code mismatch")
    snapshot_error = command(
        "snapshot",
        "restore",
        "--config",
        CONFIG,
        "--ledger",
        LEDGER,
        "--dir",
        SNAPSHOT_DIR,
        "--backup-dir",
        BACKUP_DIR,
        "--name",
        "baseline",
        "--inject-digest",
        expected=5,
    )["error"]
    check(snapshot_error["code"] == "E_SNAPSHOT", "snapshot injection code mismatch")
    divergence = command(
        "verify",
        "--config",
        CONFIG,
        "--records",
        RECORDS,
        "--model-digest",
        f"{OUT}/model_digest.txt",
        "--conclusions",
        f"{OUT}/conclusions.jsonl",
        "--group-conclusions",
        f"{OUT}/group_conclusions.jsonl",
        "--conflicts",
        f"{OUT}/conflicts.jsonl",
        "--stats",
        f"{OUT}/stats.json",
        "--inject-divergence",
        expected=6,
    )["error"]
    check(divergence["code"] == "E_DIVERGE", "divergence injection code mismatch")

    replay_path = ROOT / REPLAY_LEDGER
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / CRASH_LEDGER, replay_path)
    connection = sqlite3.connect(str(replay_path))
    try:
        row = connection.execute(
            "SELECT conclusion_json FROM conclusions ORDER BY topic, message_seq, message_id LIMIT 1"
        ).fetchone()
        conclusion = json.loads(row[0])
        conclusion["verdict"] = "at_most_once"
        connection.execute(
            "UPDATE conclusions SET verdict=?, conclusion_json=? WHERE topic=? AND message_seq=? AND message_id=?",
            (
                "at_most_once",
                json.dumps(conclusion, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                conclusion["topic"],
                conclusion["message_seq"],
                conclusion["message_id"],
            ),
        )
        connection.commit()
    finally:
        connection.close()
    replay_error = command("recover", "--ledger", REPLAY_LEDGER, "--records", RECORDS, expected=7)["error"]
    check(replay_error["code"] == "E_REPLAY", "replay injection code mismatch")

    print("[9/9] parameter evolution, stale-view replacement, and restore")
    evolved = command(
        "run",
        "--config",
        EVOLVED_CONFIG,
        "--records",
        RECORDS,
        "--ledger",
        LEDGER,
        "--out",
        EVOLVED_OUT,
    )
    check(evolved["parameter_changed"], "parameter change was not detected")
    check(evolved["parameter_version"] > base["parameter_version"], "parameter version did not increase")
    check(evolved["view_diff_counts"]["changed"] >= 1, "view diff did not report changed messages")
    command(
        "snapshot",
        "restore",
        "--config",
        CONFIG,
        "--ledger",
        LEDGER,
        "--dir",
        SNAPSHOT_DIR,
        "--backup-dir",
        BACKUP_DIR,
        "--name",
        "baseline",
    )
    final = command("recover", "--ledger", LEDGER, "--records", RECORDS)
    check(final["parameter_version"] == base["parameter_version"], "baseline parameter version was not restored")

    print(json.dumps({
        "status": "demo_pass",
        "selfcheck": selfcheck["status"],
        "model_digest": base["model_digest"],
        "conclusions_digest": base["conclusions_digest"],
        "conflicts_digest": base["conflicts_digest"],
        "stats_digest": base["stats_digest"],
        "snapshot_content_digest": saved["content_digest"],
        "parameter_version_base": base["parameter_version"],
        "parameter_version_evolved": evolved["parameter_version"],
        "negative_codes": ["E_FORMAT", "E_DUPKEY", "E_SNAPSHOT", "E_DIVERGE", "E_REPLAY"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


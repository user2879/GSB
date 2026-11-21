#!/usr/bin/env python
"""Run the local deterministic acceptance suite against samples/.

This script intentionally uses the declared loopback ports from samples/.
It never contacts a non-loopback address and creates fresh databases in a
timestamped artifact directory.
"""

from __future__ import annotations

import copy
import json
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cluster_lease.errors import LeaseError  # noqa: E402
from cluster_lease.models import ClusterSpec  # noqa: E402
from cluster_lease.orchestrator import ServiceOrchestrator  # noqa: E402
from cluster_lease.snapshots import _content_digest  # noqa: E402


def run_cli(*arguments: str, expected_exit: int = 0) -> dict[str, Any]:
    command = [sys.executable, "-m", "cluster_lease", *arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != expected_exit:
        raise AssertionError(
            f"command exited {completed.returncode}, expected {expected_exit}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    output = completed.stdout.strip()
    if not output:
        if expected_exit != 0:
            return {"code": "PROCESS_EXIT", "exit_code": completed.returncode}
        raise AssertionError(f"command emitted no JSON report: {' '.join(command)}")
    return json.loads(output)


def sample_args(scenario_dir: Path) -> list[str]:
    return [
        "--clusters",
        str(ROOT / "samples" / "clusters.json"),
        "--holders",
        str(ROOT / "samples" / "holders.json"),
        "--events",
        str(ROOT / "samples" / "events.ndjson"),
    ]


def evolved_args() -> list[str]:
    return [
        "--clusters",
        str(ROOT / "samples" / "clusters_evolved.json"),
        "--holders",
        str(ROOT / "samples" / "holders_evolved.json"),
        "--events",
        str(ROOT / "samples" / "events.ndjson"),
    ]


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "artifacts" / "acceptance" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any) -> None:
        checks.append({"details": details, "name": name, "passed": bool(passed)})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    validate = run_cli(
        "validate",
        *sample_args(output_dir),
        "--log",
        str(output_dir / "validate.jsonl"),
        "--report",
        str(output_dir / "validate.json"),
    )
    record(
        "spec validation",
        validate["ok"] and validate["event_count"] == 14 and validate["cluster_count"] == 2,
        {"event_stream_digest": validate["event_stream_digest"]},
    )

    run_a = run_cli(
        "run",
        *sample_args(output_dir),
        "--db",
        str(output_dir / "state-a.db"),
        "--runtime-dir",
        str(output_dir / "runtime-a"),
        "--batch-size",
        "4",
        "--clock-mode",
        "logical",
        "--log",
        str(output_dir / "run-a.jsonl"),
        "--report",
        str(output_dir / "run-a.json"),
    )
    run_b = run_cli(
        "run",
        *sample_args(output_dir),
        "--db",
        str(output_dir / "state-b.db"),
        "--runtime-dir",
        str(output_dir / "runtime-b"),
        "--batch-size",
        "3",
        "--clock-mode",
        "logical",
        "--log",
        str(output_dir / "run-b.jsonl"),
        "--report",
        str(output_dir / "run-b.json"),
    )
    expected = json.loads(
        (ROOT / "samples" / "expected_deterministic.json").read_text(encoding="utf-8")
    )
    deterministic = (
        run_a["reply_sequence_digest"] == run_b["reply_sequence_digest"]
        and run_a["ledger"]["ledger_content_digest"]
        == run_b["ledger"]["ledger_content_digest"]
        and run_a["decision_trace"] == run_b["decision_trace"]
        and run_a["decision_trace"] == expected["decision_sequence"]
        and run_a["reply_sequence_digest"] == expected["reply_sequence_digest"]
        and run_a["ledger"]["ledger_content_digest"]
        == expected["ledger_content_digest"]
    )
    record(
        "deterministic unique adjudication",
        deterministic and run_a["processed_event_count"] == 14,
        {
            "ledger_content_digest": run_a["ledger"]["ledger_content_digest"],
            "reply_sequence_digest": run_a["reply_sequence_digest"],
        },
    )

    database = sqlite3.connect(output_dir / "state-a.db")
    database.row_factory = sqlite3.Row
    try:
        interval_rows = database.execute(
            "SELECT cluster_id,holder_id,started_ms,ended_ms,start_event_seq,"
            "end_event_seq,end_reason FROM ownership_intervals "
            "WHERE is_stale=0 ORDER BY cluster_id,started_ms,holder_id"
        ).fetchall()
        active_counts = {
            row["cluster_id"]: row["count"]
            for row in database.execute(
                "SELECT cluster_id,COUNT(*) AS count FROM active_leases "
                "WHERE is_stale=0 GROUP BY cluster_id"
            ).fetchall()
        }
        open_counts: dict[str, int] = {}
        for row in interval_rows:
            if row["ended_ms"] is None:
                open_counts[row["cluster_id"]] = open_counts.get(row["cluster_id"], 0) + 1
            else:
                assert row["end_event_seq"] is not None and row["end_reason"] is not None
        capacity_ok = database.execute(
            "SELECT COUNT(*) FROM decisions "
            "WHERE is_stale=0 AND used_capacity > capacity"
        ).fetchone()[0] == 0
    finally:
        database.close()
    record(
        "ownership intervals and capacity invariants",
        interval_rows
        and capacity_ok
        and all(open_counts.get(cluster_id, 0) == count for cluster_id, count in active_counts.items()),
        {
            "active_counts": active_counts,
            "interval_count": len(interval_rows),
            "open_counts": open_counts,
        },
    )

    trace = {item["event_seq"]: item for item in run_a["decision_trace"]}
    expiry_ok = (
        trace[10]["cluster_id"] == "alpha"
        and trace[10]["decision"] == "granted"
        and trace[10]["active_holder_ids"] == ["h-mid"]
        and trace[11]["decision"] == "granted"
        and trace[11]["used_capacity"] == 1
    )
    record(
        "expiry reclamation and capacity constraint",
        expiry_ok,
        {"event_10": trace[10], "event_11": trace[11]},
    )

    rerun = run_cli(
        "run",
        *sample_args(output_dir),
        "--db",
        str(output_dir / "state-a.db"),
        "--runtime-dir",
        str(output_dir / "runtime-a2"),
        "--batch-size",
        "4",
        "--clock-mode",
        "logical",
        "--log",
        str(output_dir / "rerun.jsonl"),
        "--report",
        str(output_dir / "rerun.json"),
    )
    record(
        "idempotent already-committed sequence",
        rerun["processed_event_count"] == 0
        and rerun["ledger"]["ledger_content_digest"]
        == run_a["ledger"]["ledger_content_digest"],
        {"processed_event_count": rerun["processed_event_count"]},
    )

    evolve = run_cli(
        "evolve",
        *evolved_args(),
        "--db",
        str(output_dir / "state-a.db"),
        "--runtime-dir",
        str(output_dir / "runtime-evolve"),
        "--log",
        str(output_dir / "evolve.jsonl"),
        "--report",
        str(output_dir / "evolve.json"),
    )
    versions = [item["to_version"] for item in evolve.get("affected_clusters", [])]
    record(
        "whole-view parameter replacement",
        evolve.get("replacement") == "overall_replace" and versions == [2, 2],
        {
            "affected_clusters": evolve.get("affected_clusters"),
            "retained_counts": {
                key: len(value)
                for key, value in evolve.get("retained_read_only_records", {}).items()
            },
        },
    )

    snapshot_create = run_cli(
        "snapshot-create",
        *evolved_args(),
        "--db",
        str(output_dir / "state-a.db"),
        "--out",
        str(output_dir / "session.json"),
        "--log",
        str(output_dir / "snapshot-create.jsonl"),
        "--report",
        str(output_dir / "snapshot-create.json"),
    )
    snapshot_open = run_cli(
        "snapshot-open",
        *evolved_args(),
        "--db",
        str(output_dir / "state-a.db"),
        "--snapshot",
        str(output_dir / "session.json"),
        "--runtime-dir",
        str(output_dir / "runtime-snapshot"),
        "--log",
        str(output_dir / "snapshot-open.jsonl"),
        "--report",
        str(output_dir / "snapshot-open.json"),
    )
    record(
        "snapshot replay consistency",
        snapshot_open["code"] == "OK" and snapshot_open["replay"]["replay"] == "PASS",
        {
            "content_digest": snapshot_create["content_digest"],
            "snapshot_id": snapshot_create["snapshot_id"],
        },
    )

    snapshot_mismatch = run_cli(
        "snapshot-open",
        *evolved_args(),
        "--db",
        str(output_dir / "state-b.db"),
        "--snapshot",
        str(output_dir / "session.json"),
        "--runtime-dir",
        str(output_dir / "runtime-snapshot-mismatch"),
        "--log",
        str(output_dir / "snapshot-mismatch.jsonl"),
        "--report",
        str(output_dir / "snapshot-mismatch.json"),
        expected_exit=4,
    )
    record(
        "snapshot mismatch reports E_SNAPSHOT and replaces view",
        snapshot_mismatch["code"] == "E_SNAPSHOT" and snapshot_mismatch["restored"],
        {"differences": snapshot_mismatch["differences"]},
    )

    snapshot = json.loads((output_dir / "session.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(snapshot)
    first_cluster = next(item for item in tampered["clusters"] if item["cluster_id"] == "alpha")
    first_cluster["decisions"][0]["decision"] = "rejected"
    first_cluster["decisions"][0]["payload"]["conclusion"] = "rejected"
    tampered["content_digest"] = _content_digest(tampered)
    tampered_path = output_dir / "tampered-replay.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    replay_error = run_cli(
        "snapshot-open",
        *evolved_args(),
        "--db",
        str(output_dir / "state-a.db"),
        "--snapshot",
        str(tampered_path),
        "--runtime-dir",
        str(output_dir / "runtime-replay"),
        "--log",
        str(output_dir / "replay.jsonl"),
        "--report",
        str(output_dir / "replay.json"),
        expected_exit=5,
    )
    record(
        "E_REPLAY first divergence location",
        replay_error["code"] == "E_REPLAY"
        and replay_error["details"]["cluster_id"] == "alpha"
        and replay_error["details"]["event_seq"] == 1
        and replay_error["field"] == "decision",
        replay_error,
    )

    ref_error = run_cli(
        "validate",
        *sample_args(output_dir),
        "--inject",
        "missing-ref",
        "--log",
        str(output_dir / "inject-ref.jsonl"),
        "--report",
        str(output_dir / "inject-ref.json"),
        expected_exit=2,
    )
    timing_error = run_cli(
        "validate",
        *sample_args(output_dir),
        "--inject",
        "timing",
        "--log",
        str(output_dir / "inject-timing.jsonl"),
        "--report",
        str(output_dir / "inject-timing.json"),
        expected_exit=2,
    )
    snapshot_error = run_cli(
        "snapshot-open",
        *evolved_args(),
        "--db",
        str(output_dir / "state-a.db"),
        "--snapshot",
        str(output_dir / "session.json"),
        "--runtime-dir",
        str(output_dir / "runtime-inject-snapshot"),
        "--inject",
        "snapshot",
        "--log",
        str(output_dir / "inject-snapshot.jsonl"),
        "--report",
        str(output_dir / "inject-snapshot.json"),
        expected_exit=4,
    )
    record(
        "E_REF / E_TIMING / E_SNAPSHOT injections",
        ref_error["code"] == "E_REF"
        and timing_error["code"] == "E_TIMING"
        and snapshot_error["code"] == "E_SNAPSHOT",
        {"E_REF": ref_error, "E_SNAPSHOT": snapshot_error, "E_TIMING": timing_error},
    )

    def log_contains_code(path: Path, code: str) -> bool:
        return any(
            json.loads(line).get("code") == code
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    structured_logs_ok = (
        log_contains_code(output_dir / "inject-ref.jsonl", "E_REF")
        and log_contains_code(output_dir / "inject-timing.jsonl", "E_TIMING")
        and log_contains_code(output_dir / "inject-snapshot.jsonl", "E_SNAPSHOT")
    )
    record(
        "structured logs preserve business codes",
        structured_logs_ok,
        {
            "E_REF": log_contains_code(output_dir / "inject-ref.jsonl", "E_REF"),
            "E_SNAPSHOT": log_contains_code(output_dir / "inject-snapshot.jsonl", "E_SNAPSHOT"),
            "E_TIMING": log_contains_code(output_dir / "inject-timing.jsonl", "E_TIMING"),
        },
    )

    crash_db = output_dir / "state-crash.db"
    run_cli(
        "run",
        *sample_args(output_dir),
        "--db",
        str(crash_db),
        "--runtime-dir",
        str(output_dir / "runtime-crash"),
        "--batch-size",
        "4",
        "--clock-mode",
        "logical",
        "--crash-after-inflight-batch",
        "2",
        "--log",
        str(output_dir / "crash.jsonl"),
        "--report",
        str(output_dir / "crash.json"),
        expected_exit=70,
    )
    time.sleep(0.7)
    recovered = run_cli(
        "recover",
        *sample_args(output_dir),
        "--db",
        str(crash_db),
        "--runtime-dir",
        str(output_dir / "runtime-recover"),
        "--clock-mode",
        "logical",
        "--log",
        str(output_dir / "recover.jsonl"),
        "--report",
        str(output_dir / "recover.json"),
    )
    resumed = run_cli(
        "run",
        *sample_args(output_dir),
        "--db",
        str(crash_db),
        "--runtime-dir",
        str(output_dir / "runtime-resume"),
        "--batch-size",
        "4",
        "--clock-mode",
        "logical",
        "--log",
        str(output_dir / "resume.jsonl"),
        "--report",
        str(output_dir / "resume.json"),
    )
    record(
        "forced termination recovery and rerun",
        len(recovered["unknown_batches"]) == 1
        and recovered["exit_report_available"] is False
        and resumed["processed_event_sequences"] == list(range(5, 15))
        and resumed["ledger"]["ledger_content_digest"]
        == run_a["ledger"]["ledger_content_digest"],
        {
            "recovery": recovered["recovery"],
            "resumed_sequences": resumed["processed_event_sequences"],
            "unknown_batches": recovered["unknown_batches"],
        },
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 39171))
    listener.listen(1)
    orchestrator = ServiceOrchestrator(
        str(output_dir / "runtime-port-conflict"), clock_mode="logical"
    )
    try:
        try:
            orchestrator.start(
                [ClusterSpec("alpha", "127.0.0.1", 39171, 5000, 1000, 1)],
                {"alpha": {"h-mid": 5}},
                {"alpha": 1},
            )
            port_code = None
        except LeaseError as exc:
            port_code = exc.code
    finally:
        orchestrator.stop()
        listener.close()
    record("E_PORT occupied loopback port", port_code == "E_PORT", {"code": port_code})

    summary = {
        "checks": checks,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": all(check["passed"] for check in checks),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ROOT / "artifacts" / "acceptance-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

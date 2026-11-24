from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cachelease.canonical import canonical_json  # noqa: E402
from cachelease.ledger import Ledger  # noqa: E402
from cachelease.projection import build_projection, compare_projections  # noqa: E402
from cachelease.spec import load_config, load_events  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _command(args: list[str], *, expect: int = 0, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "cachelease", *args],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if expect >= 0 and result.returncode != expect:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_acceptance(work_dir: Path) -> dict[str, Any]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    sample_dir = work_dir / "sample"
    _command(["generate", "--out", str(sample_dir), "--seed", "243"])
    config_path = sample_dir / "sample.config.json"
    events_path = sample_dir / "events.ndjson"
    manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = _command(["validate", "--config", str(config_path), "--events", str(events_path)])
    port_check = _command(["port-check", "--config", str(config_path)])

    full_ledger_path = work_dir / "full.sqlite"
    full_report_path = work_dir / "full-report.json"
    _command(
        [
            "run",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--ledger",
            str(full_ledger_path),
            "--work-dir",
            str(work_dir / "full-nodes"),
            "--report",
            str(full_report_path),
        ]
    )
    full_report = _load_report(full_report_path)

    verify_report_path = work_dir / "verify-report.json"
    _command(
        [
            "verify",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--checkpoint-seq",
            str(manifest["checkpoint_seq"]),
            "--work-dir",
            str(work_dir / "verify"),
            "--report",
            str(verify_report_path),
        ],
        timeout=240,
    )
    verify_report = _load_report(verify_report_path)

    crash_ledger = work_dir / "crash.sqlite"
    crashed = _command(
        [
            "run",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--ledger",
            str(crash_ledger),
            "--work-dir",
            str(work_dir / "crash-nodes"),
            "--crash-at-batch",
            "5",
        ],
        expect=-1,
    )
    if crashed.returncode == 0:
        raise RuntimeError("crash injection unexpectedly succeeded")

    recover_report_path = work_dir / "recover-report.json"
    _command(
        [
            "recover",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--ledger",
            str(crash_ledger),
            "--work-dir",
            str(work_dir / "recover-nodes"),
            "--report",
            str(recover_report_path),
        ]
    )
    recover_report = _load_report(recover_report_path)

    resume_report_path = work_dir / "resume-report.json"
    _command(
        [
            "run",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--ledger",
            str(crash_ledger),
            "--work-dir",
            str(work_dir / "resume-nodes"),
            "--resume",
            "--report",
            str(resume_report_path),
        ]
    )
    resume_report = _load_report(resume_report_path)

    config = load_config(config_path)
    events = load_events(events_path, config)
    full_ledger = Ledger(full_ledger_path, config)
    recovered_ledger = Ledger(crash_ledger, config)
    try:
        compare_projections(
            build_projection(recovered_ledger, config),
            build_projection(full_ledger, config),
            config,
        )
        recovery_projection_equal = True
        connection = sqlite3.connect(str(full_ledger_path))
        connection.row_factory = sqlite3.Row
        outcomes = {
            int(row["seq"]): json.loads(str(row["outcome_json"]))
            for row in connection.execute("SELECT seq, outcome_json FROM events ORDER BY ordinal")
        }
        queues = [dict(row) for row in connection.execute("SELECT * FROM queue_observations")]
        queue_consistent = all(row["wait_ms"] == row["wait_end_ms"] - row["wait_start_ms"] for row in queues)
        stale_watermarks = connection.execute("SELECT COUNT(*) FROM watermarks WHERE stale=1").fetchone()[0]
        rolled_back = connection.execute(
            "SELECT COUNT(*) FROM reservations WHERE status='rolled_back'"
        ).fetchone()[0]
        arbitration = {
            "seq1_node": outcomes[1]["node_id"],
            "seq5_node": outcomes[5]["node_id"],
            "duplicate_artifact_unique_on_repeat": outcomes[1]["node_id"] == outcomes[5]["node_id"],
            "seq5_candidates": outcomes[5]["candidates"],
        }
        connection.close()
    finally:
        full_ledger.close()
        recovered_ledger.close()

    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    pytest_lines = [line for line in pytest_result.stdout.splitlines() if line.strip()]
    pytest_summary = pytest_lines[-1] if pytest_lines else ""

    summary = {
        "ok": pytest_result.returncode == 0
        and verify_report["ok"]
        and queue_consistent
        and recovery_projection_equal
        and resume_report["progress"]["last_seq"] == events[-1].seq,
        "sample": {
            "config": _display_path(config_path),
            "events": _display_path(events_path),
            "checkpoint_seq": manifest["checkpoint_seq"],
            "event_count": len(events),
            "artifacts": [artifact.key for artifact in config.artifacts],
            "nodes": [node.node_id for node in config.nodes],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "executable": sys.executable,
        },
        "validation": json.loads(validation.stdout.strip().splitlines()[-1]),
        "port_check": json.loads(port_check.stdout.strip().splitlines()[-1]),
        "full_run": {
            "progress": full_report["progress"],
            "replay": full_report["replay"],
            "content_digest": full_report["content_digest"],
            "final_watermarks": full_report["final_watermarks"],
        },
        "arbitration": arbitration,
        "reservation_rollback": {
            "seq14_status": outcomes[14]["status"],
            "seq14_error": outcomes[14]["error_code"],
            "rolled_back_rows": rolled_back,
        },
        "quota": {
            "stale_watermark_rows": stale_watermarks,
            "node_a_quota_version": full_report["final_watermarks"]["node-a"]["quota_version"],
            "node_b_quota_version": full_report["final_watermarks"]["node-b"]["quota_version"],
        },
        "scheduler": {
            "queue_observations": len(queues),
            "queue_arithmetic_consistent": queue_consistent,
            "metrics": full_report["scheduler_metrics"],
        },
        "incremental_verify": {
            "ok": verify_report["ok"],
            "checkpoint_seq": verify_report["checkpoint_seq"],
            "prefix_events": verify_report["prefix_events"],
            "suffix_events": verify_report["suffix_events"],
            "incremental_processed_events": verify_report["incremental_processed_events"],
            "prefix_snapshots_byte_identical": verify_report["prefix_snapshots_byte_identical"],
            "projection_digest": verify_report["projection_digest"],
            "progress_equal": verify_report["progress_equal"],
        },
        "recovery": {
            "crash_exit_code": crashed.returncode,
            "unknown_batches": recover_report["recovery"]["open_batches_marked_unknown"],
            "recovery_actions": recover_report["recovery_actions"],
            "replay": recover_report["replay"],
            "resume_progress": resume_report["progress"],
            "projection_equal_to_fresh_full": recovery_projection_equal,
        },
        "tests": {
            "returncode": pytest_result.returncode,
            "summary": pytest_summary,
        },
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local end-to-end acceptance suite")
    parser.add_argument("--work-dir", default=str(ROOT / "reports" / "acceptance"))
    parser.add_argument("--summary", default=str(ROOT / "reports" / "validation.json"))
    args = parser.parse_args()
    summary = run_acceptance(Path(args.work_dir))
    destination = Path(args.summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json({"ok": summary["ok"], "summary": str(destination)}))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

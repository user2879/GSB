#!/usr/bin/env python3
"""Run the complete local acceptance suite and write artifacts/verification.json."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def invoke(*args: str, timeout: int = 45, expected: set[int] | None = None) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "vecflow", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    try:
        stdout: Any = json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError:
        stdout = result.stdout[-4000:]
    record = {
        "argv": list(args),
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr_tail": result.stderr[-2000:],
    }
    if expected is not None and result.returncode not in expected:
        raise RuntimeError(f"command failed: {args}\n{json.dumps(record, ensure_ascii=False, indent=2)}")
    return record


def compact_run(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload["report"]
    schedule = report["schedule"]
    return {
        "status": report["status"],
        "run_id": report["run_id"],
        "parameter_version": report["parameter_version"],
        "content_digest": report["content_digest"],
        "plan_digest": report["plan_digest"],
        "overall_digest": report["overall_digest"],
        "shard_count": len(schedule["records"]),
        "peak_resources": schedule["peak_resources"],
        "task_peak_concurrency": schedule["task_peak_concurrency"],
        "task_results": {
            task_id: result["result_digest"] for task_id, result in sorted(report["task_results"].items())
        },
        "ledger_status": payload["ledger"]["status"],
        "ledger_format_version": payload["ledger"]["format_version"],
        "ledger_cumulative_digest": payload["ledger"]["cumulative_digest"],
    }


def compact_error(record: dict[str, Any]) -> dict[str, Any]:
    stdout = record.get("stdout") or {}
    if not isinstance(stdout, dict):
        return record
    return {
        "exit_code": record["exit_code"],
        "ok": stdout.get("ok"),
        "code": stdout.get("code"),
        "message": stdout.get("message"),
        "details": {
            key: value
            for key, value in (stdout.get("details") or {}).items()
            if key in {"task_id", "start", "end", "row_count", "block_id", "column", "expected", "actual", "global_budget", "blocked_shards", "shard_id", "field", "basis"}
        },
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "ok": False,
        "started_at": started.isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "cwd": str(ROOT),
    }

    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ.copy(),
    )
    report["unit_tests"] = {
        "exit_code": tests.returncode,
        "tail": tests.stderr[-5000:],
        "gui_skipped": "skipped 'Tkinter not available'" in tests.stderr or "skipped 'Display not available'" in tests.stderr,
    }
    if tests.returncode != 0:
        report["error"] = "unit tests failed"
        _write(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    with tempfile.TemporaryDirectory(prefix="vecflow-verify-") as td:
        root = Path(td)
        logs = root / "logs"

        report["preflight"] = invoke("preflight", "--log-dir", str(logs), expected={0})
        report["injections"] = {
            "range": compact_error(invoke("validate", "--inject-range", "--log-dir", str(logs), expected={2})),
            "nullmap": compact_error(invoke("validate", "--inject-nullmap", "--log-dir", str(logs), expected={2})),
        }
        budget = invoke(
            "run",
            "--inject-budget",
            "--ledger",
            str(root / "budget.sqlite"),
            "--snapshot-dir",
            str(root / "budget-snapshots"),
            "--log-dir",
            str(logs),
            "--reset-state",
            expected={3},
        )
        report["injections"]["budget"] = compact_error(budget)

        normal = invoke(
            "run",
            "--ledger",
            str(root / "normal.sqlite"),
            "--snapshot-dir",
            str(root / "normal-snapshots"),
            "--log-dir",
            str(logs),
            "--reset-state",
            expected={0},
        )
        report["normal_run"] = compact_run(normal["stdout"])
        normal_batch = normal["stdout"]["ledger"]["batch_id"]

        replay = invoke(
            "replay",
            "--ledger",
            str(root / "normal.sqlite"),
            "--snapshot-dir",
            str(root / "normal-snapshots"),
            "--log-dir",
            str(logs),
            "--batch-id",
            normal_batch,
            expected={0},
        )
        report["consistent_replay"] = replay["stdout"]["replay"]

        with sqlite3.connect(root / "normal.sqlite") as conn:
            first = conn.execute(
                "SELECT shard_id, result_digest FROM shard_records WHERE batch_id = ? ORDER BY task_id, shard_index LIMIT 1",
                (normal_batch,),
            ).fetchone()
            conn.execute(
                "UPDATE shard_records SET result_digest = ? WHERE batch_id = ? AND shard_id = ?",
                ("0" * 64, normal_batch, first[0]),
            )
            conn.commit()
        mismatch = invoke(
            "replay",
            "--ledger",
            str(root / "normal.sqlite"),
            "--snapshot-dir",
            str(root / "normal-snapshots"),
            "--log-dir",
            str(logs),
            "--batch-id",
            normal_batch,
            expected={4},
        )
        report["replay_mismatch"] = compact_error(mismatch)

        crash = invoke(
            "run",
            "--ledger",
            str(root / "crash.sqlite"),
            "--snapshot-dir",
            str(root / "crash-snapshots"),
            "--log-dir",
            str(logs),
            "--reset-state",
            "--force-crash-after",
            "2",
            expected={137},
        )
        recover = invoke("recover", "--ledger", str(root / "crash.sqlite"), "--log-dir", str(logs), expected={0})
        recovered_latest = recover["stdout"]["latest"]
        statuses = Counter(shard["status"] for shard in recovered_latest["shards"])
        report["forced_crash"] = {
            "exit_code": crash["exit_code"],
            "recovery": {
                "batch_id": recovered_latest["batch_id"],
                "status": recovered_latest["status"],
                "exit_report_available": recover["stdout"]["exit_report_available"],
                "exit_report": recover["stdout"]["exit_report"],
                "unknown_shards": recover["stdout"]["interrupted_batches"][0]["unknown_shards"],
                "note": recover["stdout"]["note"],
            },
            "shard_status_counts": dict(sorted(statuses.items())),
        }
        crash_replay = invoke(
            "replay",
            "--ledger",
            str(root / "crash.sqlite"),
            "--snapshot-dir",
            str(root / "crash-snapshots"),
            "--log-dir",
            str(logs),
            expected={0},
        )
        report["crash_replay"] = crash_replay["stdout"]["replay"]

        preflight_stdout = report["preflight"]["stdout"]
        if preflight_stdout.get("gui_ready"):
            gui = invoke(
                "gui",
                "--smoke",
                "--smoke-ms",
                "120",
                "--ledger",
                str(root / "gui.sqlite"),
                "--snapshot-dir",
                str(root / "gui-snapshots"),
                "--log-dir",
                str(logs),
                expected={0},
            )
            report["gui_smoke"] = gui["stdout"]
        else:
            report["gui_smoke"] = {
                "ok": True,
                "skipped": True,
                "reason": "tkinter_available or display unavailable in this shell",
                "tkinter_available": preflight_stdout.get("tkinter_available"),
                "display": preflight_stdout.get("display"),
            }
        log_codes = Counter()
        log_path = logs / "vecflow.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("code"):
                    log_codes[event["code"]] += 1
        report["log_codes"] = dict(sorted(log_codes.items()))
        report["log_lines_valid_json"] = True

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["ok"] = (
        report["normal_run"]["status"] == "committed"
        and report["consistent_replay"]["compared_shards"] == report["normal_run"]["shard_count"]
        and report["replay_mismatch"]["code"] == "E_REPLAY"
        and report["forced_crash"]["recovery"]["exit_report_available"] is False
        and report["crash_replay"]["unknown_rerun_shards"]
        and report["gui_smoke"]["ok"] is True
        and all(code in report["log_codes"] for code in ("E_RANGE", "E_NULLMAP", "E_BUDGET", "E_REPLAY"))
    )
    _write(report)
    print(json.dumps({key: report[key] for key in ("ok", "normal_run", "consistent_replay", "forced_crash", "crash_replay", "gui_smoke", "log_codes")}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def _write(report: dict[str, Any]) -> None:
    target = ROOT / "artifacts" / "verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

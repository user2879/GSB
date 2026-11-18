from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANOMALY = ROOT / "var" / "anomaly"
DATA = ROOT / "data"
PARAMS = DATA / "params"


def reset_directory() -> None:
    allowed_parent = (ROOT / "var").resolve()
    if ANOMALY.exists():
        resolved = ANOMALY.resolve()
        if resolved.parent != allowed_parent or resolved.name != "anomaly":
            raise RuntimeError(f"refusing to remove unexpected path: {resolved}")
        shutil.rmtree(ANOMALY)
    ANOMALY.mkdir(parents=True, exist_ok=False)


def run_cli(arguments: list[str], expected_exit: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "columnar_pipeline", *arguments]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != expected_exit:
        raise RuntimeError(
            f"command returned {completed.returncode}, expected {expected_exit}: "
            f"{' '.join(command)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON output: {' '.join(command)}")
    return json.loads(lines[-1])


def main() -> int:
    reset_directory()
    ledger = ANOMALY / "ledger.sqlite"
    snapshots = ANOMALY / "snapshots"
    log_file = ANOMALY / "pipeline.jsonl"
    checks: list[dict[str, Any]] = []
    injections = [
        ("E_SKEW", DATA / "skew_amplified.json", PARAMS / "inject_skew.json", 20),
        ("E_BUDGET", DATA / "small.json", PARAMS / "inject_budget.json", 30),
        ("E_DIGEST", DATA / "small.json", PARAMS / "inject_digest.json", 40),
    ]
    for code, data_path, config_path, exit_code in injections:
        payload = run_cli(
            [
                "run",
                "--data",
                str(data_path),
                "--config",
                str(config_path),
                "--out-dir",
                str(ANOMALY / code.lower()),
                "--ledger",
                str(ledger),
                "--snapshot-dir",
                str(snapshots),
                "--log-file",
                str(log_file),
            ],
            expected_exit=exit_code,
        )
        checks.append(
            {
                "code": payload["code"],
                "expected_code": code,
                "exit_code": exit_code,
                "task_id": payload.get("task_id"),
                "location": payload.get("location"),
            }
        )

    baseline = run_cli(
        [
            "run",
            "--data",
            str(DATA / "small.json"),
            "--config",
            str(PARAMS / "small.json"),
            "--out-dir",
            str(ANOMALY / "replay_baseline"),
            "--ledger",
            str(ledger),
            "--snapshot-dir",
            str(snapshots),
            "--log-file",
            str(log_file),
        ],
        expected_exit=0,
    )
    with sqlite3.connect(ledger) as connection:
        row = connection.execute(
            "SELECT task_id, rows_completed FROM tasks WHERE run_id = ? ORDER BY replay_order LIMIT 1",
            (baseline["run_id"],),
        ).fetchone()
        task_id, rows_completed = row
        connection.execute(
            "UPDATE tasks SET rows_completed = ? WHERE run_id = ? AND task_id = ?",
            (rows_completed + 1, baseline["run_id"], task_id),
        )
        connection.commit()
    replay_payload = run_cli(
        [
            "replay",
            "--snapshot",
            baseline["snapshot_path"],
            "--data",
            str(DATA / "small.json"),
            "--ledger",
            str(ledger),
            "--out-dir",
            str(ANOMALY / "replay_mismatch"),
            "--log-file",
            str(log_file),
        ],
        expected_exit=50,
    )
    checks.append(
        {
            "code": replay_payload["code"],
            "expected_code": "E_REPLAY",
            "exit_code": 50,
            "task_id": replay_payload.get("task_id"),
            "field": replay_payload.get("field"),
        }
    )
    summary = {
        "format_version": 1,
        "checks": checks,
        "all_codes_correct": [item["code"] for item in checks]
        == ["E_SKEW", "E_BUDGET", "E_DIGEST", "E_REPLAY"],
        "structured_log": str(log_file),
    }
    (ANOMALY / "anomaly_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if summary["all_codes_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the local acceptance suite and write a consolidated report."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
WORK = ROOT / ".workbench" / "acceptance"
PYTHON = sys.executable


def run_cli(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    command = [PYTHON, "-m", "news_diag", *args]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != expect:
        raise RuntimeError(
            f"command failed ({completed.returncode} != {expect}): {' '.join(command)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    run_cli("run", "--clean-snapshots", "--snapshot-dir", str(WORK / "full-snapshots"),
            "--report", str(ARTIFACTS / "report.json"),
            "--output", str(ARTIFACTS / "output.json"))
    full_report = read_json(ARTIFACTS / "report.json")
    full_output = read_json(ARTIFACTS / "output.json")

    run_cli("run", "--snapshot-dir", str(WORK / "full-snapshots"),
            "--report", str(ARTIFACTS / "incremental_report.json"),
            "--output", str(ARTIFACTS / "incremental_output.json"))
    incremental_report = read_json(ARTIFACTS / "incremental_report.json")
    incremental_equal = read_json(ARTIFACTS / "incremental_output.json") == full_output

    run_cli("verify", "--work-dir", str(WORK / "verify"),
            "--report", str(ARTIFACTS / "equivalence_report.json"),
            "--output", str(ARTIFACTS / "verified_output.json"))
    equivalence = read_json(ARTIFACTS / "equivalence_report.json")

    run_cli("run", "--injections", "samples/injections.yaml", "--clean-snapshots",
            "--snapshot-dir", str(WORK / "injection-snapshots"),
            "--report", str(ARTIFACTS / "injection_report.json"),
            "--output", str(ARTIFACTS / "injection_output.json"))
    injection_report = read_json(ARTIFACTS / "injection_report.json")

    signal_snapshot = WORK / "signal-snapshots"
    run_cli("run", "--snapshot-dir", str(signal_snapshot),
            "--report", str(ARTIFACTS / "signal_partial_report.json"),
            "--output", str(ARTIFACTS / "signal_partial_output.json"),
            "--stop-after-docs", "2", "--clean-snapshots")
    signal_partial = read_json(ARTIFACTS / "signal_partial_report.json")
    run_cli("run", "--snapshot-dir", str(signal_snapshot),
            "--report", str(ARTIFACTS / "signal_resume_report.json"),
            "--output", str(ARTIFACTS / "signal_resume_output.json"))
    signal_resume = read_json(ARTIFACTS / "signal_resume_report.json")
    signal_equal = read_json(ARTIFACTS / "signal_resume_output.json") == full_output

    os_signal_snapshot = WORK / "os-signal-snapshots"
    os_signal_command = [
        PYTHON, "-m", "news_diag", "run",
        "--snapshot-dir", str(os_signal_snapshot),
        "--report", str(ARTIFACTS / "sigterm_report.json"),
        "--output", str(ARTIFACTS / "sigterm_output.json"),
        "--doc-delay-ms", "200", "--clean-snapshots",
    ]
    process = subprocess.Popen(os_signal_command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _wait_for_current(os_signal_snapshot)
    process.send_signal(signal.SIGTERM)
    os_signal_stdout, os_signal_stderr = process.communicate(timeout=20)
    os_signal_report = read_json(ARTIFACTS / "sigterm_report.json")

    kill_snapshot = WORK / "kill-snapshots"
    kill_command = [
        PYTHON, "-m", "news_diag", "run",
        "--snapshot-dir", str(kill_snapshot),
        "--report", str(ARTIFACTS / "kill_partial_report.json"),
        "--output", str(ARTIFACTS / "kill_partial_output.json"),
        "--doc-delay-ms", "200", "--clean-snapshots",
    ]
    killed = subprocess.Popen(kill_command, cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    committed_name = _wait_for_current(kill_snapshot)
    os.kill(killed.pid, signal.SIGKILL)
    killed.communicate(timeout=20)
    run_cli("run", "--snapshot-dir", str(kill_snapshot),
            "--report", str(ARTIFACTS / "kill_resume_report.json"),
            "--output", str(ARTIFACTS / "kill_resume_output.json"))
    kill_resume = read_json(ARTIFACTS / "kill_resume_report.json")
    kill_equal = read_json(ARTIFACTS / "kill_resume_output.json") == full_output

    stale_snapshot = WORK / "stale-snapshots"
    run_cli("run", "--clean-snapshots", "--snapshot-dir", str(stale_snapshot),
            "--report", str(ARTIFACTS / "stale_seed_report.json"),
            "--output", str(ARTIFACTS / "stale_seed_output.json"))
    run_cli("run", "--snapshot-dir", str(stale_snapshot), "--expect-version", "999999",
            "--report", str(ARTIFACTS / "stale_report.json"),
            "--output", str(ARTIFACTS / "stale_output.json"))
    stale_report = read_json(ARTIFACTS / "stale_report.json")

    test_process = subprocess.run([PYTHON, "-m", "pytest", "-q"], cwd=ROOT,
                                  text=True, capture_output=True)

    summary = {
        "schema_version": 1,
        "acceptance_passed": all([
            full_report["summary"]["document_count"] == 10,
            full_report["failure_count"] == 1,
            incremental_equal,
            equivalence["all_equal"],
            injection_report["injection_hit_count"] == 5,
            signal_equal,
            process.returncode == 0 and os_signal_report.get("handled_signal") == signal.SIGTERM,
            killed.returncode == -signal.SIGKILL,
            committed_name.startswith("gen-"),
            kill_equal,
            stale_report.get("stale") is True,
            any(item["code"] == "E_SNAPSHOT_STALE" for item in stale_report.get("failures", [])),
            test_process.returncode == 0,
        ]),
        "full_run": {
            "mode": full_report["mode"],
            "snapshot_version": full_report["snapshot_version"],
            "summary": full_report["summary"],
            "failure_count": full_report["failure_count"],
            "failures": full_report["failures"],
            "reuse_counts": full_report["reuse_counts"],
            "canonical_digest": full_report["canonical_digest"],
        },
        "incremental_noop": {
            "mode": incremental_report["mode"],
            "reuse_counts": incremental_report["reuse_counts"],
            "canonical_equal_to_full": incremental_equal,
        },
        "incremental_vs_full_per_command": {
            "all_equal": equivalence["all_equal"],
            "steps_verified": equivalence["steps_verified"],
            "steps": equivalence["steps"],
        },
        "fault_injection": {
            "hit_count": injection_report["injection_hit_count"],
            "all_code_match": all(item.get("code_match") for item in injection_report["injections"]),
            "injections": injection_report["injections"],
            "summary": injection_report["summary"],
        },
        "graceful_signal": {
            "simulated_exit_code": 0,
            "simulated_interrupted": signal_partial["interrupted"],
            "simulated_completed_documents": signal_partial["completed_document_ids"],
            "resume_mode": signal_resume["mode"],
            "resume_reused": signal_resume["reuse_counts"]["reused"],
            "resume_equal_to_full": signal_equal,
            "os_sigterm_exit_code": process.returncode,
            "os_sigterm_handled": os_signal_report.get("handled_signal"),
            "os_sigterm_interrupted": os_signal_report.get("interrupted"),
            "os_sigterm_stdout": os_signal_stdout.strip(),
            "os_sigterm_stderr": os_signal_stderr.strip(),
        },
        "forced_kill_recovery": {
            "killed_exit_code": killed.returncode,
            "committed_generation_at_kill": committed_name,
            "resume_mode": kill_resume["mode"],
            "resume_reused": kill_resume["reuse_counts"]["reused"],
            "resume_equal_to_full": kill_equal,
        },
        "stale_snapshot": {
            "stale": stale_report.get("stale"),
            "reuse_counts": stale_report.get("reuse_counts"),
            "errors": [item for item in stale_report.get("failures", [])
                       if item.get("code") == "E_SNAPSHOT_STALE"],
        },
        "tests": {
            "return_code": test_process.returncode,
            "stdout": test_process.stdout.strip(),
            "stderr": test_process.stderr.strip(),
        },
    }
    (ARTIFACTS / "acceptance_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "acceptance_passed": summary["acceptance_passed"],
        "report": "artifacts/acceptance_report.json",
    }, ensure_ascii=False))
    return 0 if summary["acceptance_passed"] else 1


def _wait_for_current(snapshot_dir: Path, timeout: float = 10.0) -> str:
    pointer = snapshot_dir / "CURRENT"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pointer.exists():
            return pointer.read_text(encoding="utf-8").strip()
        time.sleep(0.02)
    raise RuntimeError(f"snapshot CURRENT was not committed: {snapshot_dir}")


if __name__ == "__main__":
    raise SystemExit(main())


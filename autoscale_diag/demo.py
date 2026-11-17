"""Deterministic three-scenario local demonstration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .reporting import build_report, write_json_atomic


def _json_from_stdout(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON: {completed.args}")
    return json.loads(lines[-1])


def _json_from_stderr(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in completed.stderr.splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def _run_cli(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(root)
        if not existing_python_path
        else str(root) + os.pathsep + existing_python_path
    )
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "autoscale_diag", *args],
        cwd=root,
        env=environment,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _clean_demo_output(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("workbench.db", "workbench.db-wal", "workbench.db-shm", "workbench.db.bak"):
        path = directory / name
        if path.exists():
            path.unlink()
    for name in ("v1", "v2"):
        path = directory / name
        if path.exists():
            shutil.rmtree(path)


def run_demo(directory: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    out = directory.resolve()
    _clean_demo_output(out)
    db = out / "workbench.db"
    spec_v1 = root / "samples" / "demo_cluster.json"
    spec_v2 = root / "samples" / "demo_cluster_budget_v2.json"
    run_id = "demo"

    cold = _run_cli(
        root,
        [
            "run",
            str(spec_v1),
            "--db",
            str(db),
            "--run-id",
            run_id,
            "--stop-after",
            "4",
        ],
    )
    if cold.returncode != 137:
        raise RuntimeError(
            "forced termination simulation did not exit with 137: "
            f"rc={cold.returncode} stderr={cold.stderr}"
        )
    with Ledger(db) as ledger:
        interrupted_run = ledger.get_run(run_id)
        prefix = ledger.load_committed(run_id)
        cold_payload = {
            "scenario": "01_cold_start_and_forced_termination",
            "forced_exit_code": cold.returncode,
            "status": interrupted_run["status"] if interrupted_run else None,
            "committed_minutes": len(prefix),
            "minute_count": interrupted_run["minute_count"] if interrupted_run else None,
            "exit_report_available": False,
            "decisions": [
                {
                    "minute": header["minute"],
                    "budget_used": header["budget_used"],
                    "decisions": [
                        {
                            "workload_id": item["workload_id"],
                            "expected_replicas": item["expected_replicas"],
                            "final_replicas": item["final_replicas"],
                            "decision_type": item["decision_type"],
                        }
                        for item in header["decisions"]
                    ],
                }
                for header in prefix
            ],
            "note": "only fully committed minute transactions are visible",
        }
    write_json_atomic(out / "01_cold_start.json", cold_payload)

    resume = _run_cli(
        root,
        [
            "resume",
            str(spec_v1),
            "--db",
            str(db),
            "--run-id",
            run_id,
            "--out-dir",
            str(out / "v1"),
        ],
    )
    if resume.returncode != 0:
        raise RuntimeError(f"resume failed: rc={resume.returncode} stderr={resume.stderr}")
    resume_payload = _json_from_stdout(resume)
    replay = _run_cli(
        root,
        ["replay", str(spec_v1), "--db", str(db), "--run-id", run_id],
    )
    if replay.returncode != 0:
        raise RuntimeError(f"v1 replay failed: rc={replay.returncode} stderr={replay.stderr}")
    replay_payload = _json_from_stdout(replay)
    with Ledger(db) as ledger:
        report_v1 = build_report(ledger, run_id)
        recovery = ledger.recovery_events(run_id)
        suppressions = ledger.cooling_suppressions(run_id)
    cooling_by_minute = {}
    for item in suppressions:
        cooling_by_minute.setdefault(item["minute_index"], []).append(item)
    cooldown_payload = {
        "scenario": "02_cooldown_suppression_after_expansion_and_recovery_replay",
        "run": _report_summary(report_v1),
        "recovery_events": recovery,
        "cooldown_suppressions": suppressions,
        "replay": replay_payload,
        "minutes_with_suppression": sorted(cooling_by_minute),
        "report_files": resume_payload.get("report_files", {}),
    }
    write_json_atomic(out / "02_cooldown_suppression.json", cooldown_payload)

    replacement = _run_cli(
        root,
        [
            "run",
            str(spec_v2),
            "--db",
            str(db),
            "--run-id",
            run_id,
            "--replace",
            "--out-dir",
            str(out / "v2"),
        ],
    )
    if replacement.returncode != 0:
        raise RuntimeError(
            f"budget replacement failed: rc={replacement.returncode} stderr={replacement.stderr}"
        )
    replacement_payload = _json_from_stdout(replacement)
    with Ledger(db) as ledger:
        report_v2 = build_report(ledger, run_id)
    replace_payload = {
        "scenario": "03_budget_change_whole_run_replacement",
        "run": _report_summary(report_v2),
        "replacement_backup": replacement_payload.get("replacement_backup"),
        "report_files": replacement_payload.get("report_files", {}),
        "budget_before": report_v1["run"]["budget"],
        "budget_after": report_v2["run"]["budget"],
        "path_before": report_v1["run"]["path"],
        "path_after": report_v2["run"]["path"],
    }
    write_json_atomic(out / "03_budget_replacement.json", replace_payload)

    parity = _run_cli(root, ["compare-paths", str(spec_v1)])
    if parity.returncode != 0:
        raise RuntimeError(f"path parity failed: rc={parity.returncode} stderr={parity.stderr}")
    parity_payload = _json_from_stdout(parity)
    write_json_atomic(out / "04_path_parity.json", parity_payload)

    replay_guard = _run_cli(
        root,
        ["replay", str(spec_v1), "--db", str(db), "--run-id", run_id],
    )
    guard_payload = {
        "scenario": "05_replay_guard_after_declaration_replacement",
        "expected_exit_code": 3,
        "actual_exit_code": replay_guard.returncode,
        "business_error": _json_from_stderr(replay_guard),
        "replacement_declaration_version": report_v2["run"]["declaration_version"],
        "old_declaration_version": report_v1["run"]["declaration_version"],
    }
    write_json_atomic(out / "05_replay_guard.json", guard_payload)

    ui_check = _run_cli(
        root,
        ["ui", "--db", str(db), "--run-id", run_id, "--check"],
    )
    ui_payload = _json_from_stdout(ui_check) if ui_check.stdout.strip() else {}
    write_json_atomic(out / "06_ui_viewmodel_check.json", ui_payload)

    checks = {
        "forced_termination_committed_prefix": cold_payload["committed_minutes"] == 4,
        "resume_completed_run": report_v1["run"]["status"] == "complete",
        "cooldown_suppression_observed": len(suppressions) > 0,
        "replay_verified_all_minutes": replay_payload.get("verified_minutes") == 12,
        "budget_replacement_complete": report_v2["run"]["status"] == "complete",
        "budget_changed": report_v1["run"]["budget"] != report_v2["run"]["budget"],
        "path_before_full": report_v1["run"]["path"] == "full",
        "path_after_incremental": report_v2["run"]["path"] == "incremental",
        "path_parity": parity_payload.get("byte_identical_decisions_and_metrics") is True,
        "version_mismatch_guard": replay_guard.returncode == 3
        and _json_from_stderr(replay_guard).get("business_code") == "E_REPLAY",
        "ui_viewmodel_check": ui_payload.get("ok") is True,
    }
    summary = {
        "ok": all(checks.values()),
        "business_codes": ["OK", "E_COOLDOWN"] if suppressions else ["OK"],
        "checks": checks,
        "scenario_files": sorted(path.name for path in out.glob("0*.json")),
        "ledger_digest_v1": report_v1["ledger_digest"],
        "ledger_digest_v2": report_v2["ledger_digest"],
        "path_parity_digest": parity_payload.get("full", {}).get("decision_digest"),
        "ui_tkinter_available": ui_payload.get("tkinter_available"),
        "output_directory": str(out),
    }
    write_json_atomic(out / "summary.json", summary)
    return summary


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run"]["run_id"],
        "declaration_version": report["run"]["declaration_version"],
        "status": report["run"]["status"],
        "path": report["run"]["path"],
        "window_capacity": report["run"]["window_capacity"],
        "peak_memory_estimate_bytes": report["run"]["peak_memory_estimate_bytes"],
        "committed_minutes": report["run"]["committed_minutes"],
        "minute_count": report["run"]["minute_count"],
        "budget": report["run"]["budget"],
        "summary": report["summary"],
        "ledger_digest": report["ledger_digest"],
        "business_codes": report["business_codes"],
        "exit_report_available": report["exit_report_available"],
    }

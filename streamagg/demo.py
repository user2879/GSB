from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_utils import JsonlLogger
from .scenarios import DemoPaths


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _run_child(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(cwd) + (os.pathsep + existing if existing else "")
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    parsed = _parse_last_json(process.stdout)
    return {
        "process_exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "parsed": parsed,
    }


def run_demo(workspace: str | Path, *, clean: bool = True) -> dict[str, Any]:
    workspace_path = Path(workspace).resolve()
    demo_root = workspace_path / "artifacts" / "demo"
    if clean and demo_root.exists():
        shutil.rmtree(demo_root)
    paths = DemoPaths.from_values(workspace_path, demo_root)
    paths.prepare()
    logger = JsonlLogger(paths.logs, "demo")
    logger.info(
        "OK",
        "demo_start",
        "three-scenario demonstration started",
        {"workspace": str(workspace_path), "ledger": str(paths.ledger)},
    )
    common = [
        "--workspace",
        str(paths.workspace),
        "--demo-root",
        str(paths.demo_root),
        "--ledger",
        str(paths.ledger),
        "--snapshots",
        str(paths.snapshots),
        "--logs",
        str(paths.logs),
        "--reports",
        str(paths.reports),
    ]
    scenarios: list[dict[str, Any]] = []
    expected_scenarios = {
        1: "OK_SCENARIO_1_ORDERED",
        2: "OK_SCENARIO_2_LATE_BUDGET",
        3: "OK_SCENARIO_3_REPLACE_SNAPSHOT_REPLAY",
    }
    for number in (1, 2, 3):
        command = [
            sys.executable,
            "-m",
            "streamagg.cli",
            "scenario",
            "--scenario",
            str(number),
            *common,
        ]
        child = _run_child(
            command,
            cwd=workspace_path,
            stdout_path=paths.reports / f"scenario{number}.stdout.json",
            stderr_path=paths.reports / f"scenario{number}.stderr.log",
        )
        report_path = paths.reports / f"scenario{number}.json"
        report: dict[str, Any] = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        business_code = (child["parsed"] or {}).get("business_code") or report.get("business_code")
        passed = (
            child["process_exit_code"] == 0
            and business_code == expected_scenarios[number]
        )
        scenarios.append(
            {
                "scenario": number,
                "expected_business_code": expected_scenarios[number],
                "business_code": business_code,
                "process_exit_code": child["process_exit_code"],
                "passed": passed,
                "report_path": str(report_path),
                "stdout_path": str(paths.reports / f"scenario{number}.stdout.json"),
                "stderr_path": str(paths.reports / f"scenario{number}.stderr.log"),
            }
        )

    injections: list[dict[str, Any]] = []
    expected_injections = {
        "order": ("E_ORDER", 3),
        "grid": ("E_GRID", 4),
        "snapshot": ("E_SNAPSHOT", 5),
        "replay": ("E_REPLAY", 6),
    }
    for kind, (expected_code, expected_exit) in expected_injections.items():
        command = [
            sys.executable,
            "-m",
            "streamagg.cli",
            "inject",
            "--kind",
            kind,
            *common,
        ]
        child = _run_child(
            command,
            cwd=workspace_path,
            stdout_path=paths.reports / f"injection_{kind}.stdout.json",
            stderr_path=paths.reports / f"injection_{kind}.stderr.log",
        )
        parsed = child["parsed"] or {}
        actual_code = parsed.get("code")
        passed = child["process_exit_code"] == expected_exit and actual_code == expected_code
        injections.append(
            {
                "kind": kind,
                "expected_business_code": expected_code,
                "business_code": actual_code,
                "expected_exit_code": expected_exit,
                "process_exit_code": child["process_exit_code"],
                "passed": passed,
                "structured_log": str(paths.logs / f"injection_{kind}.jsonl"),
            }
        )

    all_passed = all(item["passed"] for item in scenarios + injections)
    report = {
        "format_version": 1,
        "generated_at": _now(),
        "workspace": str(workspace_path),
        "ledger": str(paths.ledger),
        "snapshots": str(paths.snapshots),
        "logs": str(paths.logs),
        "scenarios": scenarios,
        "injections": injections,
        "all_passed": all_passed,
        "business_code": "OK_DEMO" if all_passed else "E_DEMO_VERIFY",
    }
    report_path = paths.reports / "demo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    if all_passed:
        logger.info("OK", "demo_complete", "all scenarios and injections passed", {"report": str(report_path)})
    else:
        logger.emit(
            level="ERROR",
            code="E_DEMO_VERIFY",
            event="demo_complete",
            message="one or more scenarios or injections failed",
            details={"report": str(report_path)},
        )
    return report

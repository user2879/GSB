#!/usr/bin/env python3
"""Run the local test suite and repeatable demo, then record actual results."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DEMO_WORKSPACE = ROOT / "demo-workspace"


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    version = run([sys.executable, "--version"])
    tests = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    demo = run(
        [
            sys.executable,
            "-m",
            "cacheproof",
            "demo",
            "--workspace",
            str(DEMO_WORKSPACE),
            "--reset",
            "--report",
            str(ARTIFACTS / "validation-results.json"),
            "--json",
        ],
        timeout=180,
    )
    demo_json = json.loads(demo.stdout)
    tkinter_check = run(
        [
            sys.executable,
            "-c",
            "import tkinter; print(tkinter.TkVersion)",
        ]
    )
    summary = {
        "status": "PASS" if tests.returncode == 0 and demo.returncode == 0 and demo_json.get("ok") else "FAIL",
        "python": (version.stdout or version.stderr).strip(),
        "platform": platform.platform(),
        "tests_exit_code": tests.returncode,
        "tests_output_tail": (tests.stderr or tests.stdout).strip().splitlines()[-5:],
        "demo_exit_code": demo.returncode,
        "demo_report": "artifacts/validation-results.json",
        "demo_phases": [
            {
                "phase": phase.get("phase"),
                "code": phase.get("code"),
                "ok": phase.get("ok", True),
            }
            for phase in demo_json.get("phases", [])
        ],
        "tkinter_check_exit_code": tkinter_check.returncode,
        "tkinter": (tkinter_check.stdout or tkinter_check.stderr).strip(),
    }
    summary_path = ARTIFACTS / "validation-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

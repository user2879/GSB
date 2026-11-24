from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_run_and_snapshot_info(sample_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "routing_workbench",
            "run",
            "--samples",
            str(sample_dir),
            "--state",
            str(tmp_path / "state"),
            "--report",
            str(report_path),
            "--batch-budget",
            "1000000",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["result_count"] == 13
    assert report["summary"]["failure_count"] == 0
    info = subprocess.run(
        [sys.executable, "-m", "routing_workbench", "snapshot-info", "--state", str(tmp_path / "state")],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert info.returncode == 0
    manifest = json.loads(info.stdout)
    assert manifest["completed_command_cursor"] == 12
    assert manifest["format_version"] == 1


from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from remix_workbench.engine import RemixEngine


def test_graceful_stage_boundary_stop_commits_and_resumes(
    config: dict,
    samples_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    workdir = tmp_path / "graceful"
    monkeypatch.setenv("RMXW_GRACEFUL_STOP_AFTER_STAGE", "parse")
    stopped = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        force_workers=4,
    ).execute()
    assert stopped.stopped is True
    assert stopped.exit_code == 0
    assert stopped.report is None
    progress = json.loads((workdir / "progress.json").read_text(encoding="utf-8"))
    assert progress["stopped"] is True
    assert progress["stage"] == "parse"
    assert progress["stage_versions"]["parse"] > 0
    assert not (workdir / "report.json").exists()
    monkeypatch.delenv("RMXW_GRACEFUL_STOP_AFTER_STAGE")
    resumed = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        force_workers=4,
    ).execute()
    assert resumed.report is not None
    assert resumed.report["recovery"]["used"] is True
    assert resumed.report["recovery"]["stage_versions"]["parse"] > 0
    assert resumed.report["cache_summary"]["reuse_count"] > 0


def test_forced_kill_leaves_no_report_and_recovers_only_complete_snapshot(
    config: dict,
    samples_root: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "killed"
    report_path = tmp_path / "killed_report.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["RMXW_FORCE_KILL_AFTER_STAGE"] = "parse"
    command = [
        sys.executable,
        "-m",
        "remix_workbench",
        "run",
        "--samples-root",
        str(samples_root),
        "--config",
        str(workspace / "configs" / "remix.json"),
        "--workdir",
        str(workdir),
        "--report",
        str(report_path),
        "--skip-serial-compare",
    ]
    process = subprocess.run(command, env=env, capture_output=True, text=True)
    assert process.returncode == 137
    assert not report_path.exists()
    actual_report = workdir / "report.json"
    assert not actual_report.exists()
    current = json.loads((workdir / "snapshots" / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["global_version"] >= 1
    resumed = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        force_workers=2,
    ).execute()
    assert resumed.report is not None
    assert resumed.report["recovery"]["used"] is True
    assert resumed.report["recovery"]["snapshot_version"] >= 1
    assert resumed.report["recovery"]["stage_versions"]["parse"] >= 1
    assert resumed.report["recovery"]["completed_fragments"] == []


def test_sigterm_finishes_stage_and_exits_zero_without_report(
    samples_root: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "sigterm"
    report_path = tmp_path / "sigterm_report.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["RMXW_STAGE_DELAY_SECONDS"] = "0.4"
    command = [
        sys.executable,
        "-m",
        "remix_workbench",
        "run",
        "--samples-root",
        str(samples_root),
        "--config",
        str(workspace / "configs" / "remix.json"),
        "--workdir",
        str(workdir),
        "--report",
        str(report_path),
        "--skip-serial-compare",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    progress_path = workdir / "progress.json"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("global_version", 0) >= 1:
                break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    assert process.poll() is None, "process finished before SIGTERM could be delivered"
    process.terminate()
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, (stdout, stderr)
    assert not report_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["stopped"] is True
    assert progress["received_signal"] == 15

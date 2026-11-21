from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import ROOT, clone, write_config


def _run_cli(args: list[str], timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "layerpack", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_sigterm_finishes_current_stage_and_exits_zero(
    sample_paths, base_config, tmp_path
):
    config = clone(base_config)
    config["runtime"]["stage_delay_ms"] = 1200
    config_path = write_config(tmp_path / "signal.yaml", config)
    state_dir = tmp_path / "state"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "layerpack",
            "run",
            "--weights",
            str(sample_paths["weights"]),
            "--calibration",
            str(sample_paths["calibration"]),
            "--config",
            str(config_path),
            "--state-dir",
            str(state_dir),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.8)
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr
    report = json.loads(stdout)
    assert report["status"] == "interrupted"
    assert report["exit_code"] == 0
    assert report["restore_point"] is not None
    assert "loading" in report["restore_point"]["phases"]


def test_forced_kill_recovers_only_complete_snapshot(sample_paths, base_config, tmp_path):
    state_dir = tmp_path / "state"
    first = _run_cli(
        [
            "run",
            "--weights",
            str(sample_paths["weights"]),
            "--calibration",
            str(sample_paths["calibration"]),
            "--config",
            str(sample_paths["config"]),
            "--state-dir",
            str(state_dir),
        ]
    )
    assert first.returncode == 0, first.stderr
    first_report = json.loads(first.stdout)
    complete_version = first_report["snapshot_version"]

    changed = clone(base_config)
    changed["layers"]["head"]["bits"] = 16
    changed["runtime"]["stage_delay_ms"] = 1000
    changed_path = write_config(tmp_path / "changed.yaml", changed)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "layerpack",
            "run",
            "--weights",
            str(sample_paths["weights"]),
            "--calibration",
            str(sample_paths["calibration"]),
            "--config",
            str(changed_path),
            "--state-dir",
            str(state_dir),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.25)
    process.kill()
    process.communicate(timeout=10)
    status = _run_cli(["status", "--state-dir", str(state_dir)])
    assert status.returncode == 0, status.stderr
    recovered = json.loads(status.stdout)
    assert recovered["restore_point"] is not None
    assert recovered["restore_point"]["snapshot_version"] == complete_version
    assert recovered["restore_point"]["status"] == "complete"

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from vc_workbench.state import load_current

from conftest import SAMPLES, run_cli


def test_budget_exhaustion_preserves_todo_and_partial_assembly(tmp_path: Path) -> None:
    config = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
    config["speech_frame_budget"] = 70
    config_path = tmp_path / "budget_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report_path = tmp_path / "report.json"
    result = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        config_path,
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report_path,
        "--fresh",
        "--stop-after-command",
        "1",
    )
    assert result.returncode == 4
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["budget"]["consumed"] == 49
    assert report["budget"]["remaining"] == 21
    assert report["budget"]["replay_ok"] is True
    assert report["budget"]["todo"] == ["seg_002", "seg_003"]
    assert report["budget"]["todo_cursor"] == 1
    assert report["budget"]["todo_replay_ok"] is True
    assert [failure["code"] for failure in report["failures"]] == [
        "E_FRAME_BUDGET_EXHAUSTED",
        "E_FRAME_BUDGET_EXHAUSTED",
    ]
    assert report["assembly"]["status"] == "partial"
    assert [item["segment_id"] for item in report["assembly"]["included"]] == ["seg_001"]
    assert report["assembly"]["sample_count"] == 8000

    second = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        config_path,
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report_path,
        "--stop-after-command",
        "2",
    )
    assert second.returncode == 4
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["budget"]["consumed"] == 49
    assert len(report["budget"]["charges"]) == 1
    assert report["budget"]["replay_ok"] is True


def test_forced_kill_after_command_and_recovery(baseline: dict[str, object], tmp_path: Path) -> None:
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report = tmp_path / "report.json"
    crashed = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
        "--fresh",
        "--crash-after-command",
        "3",
    )
    assert crashed.returncode == 137
    assert not report.exists()
    point = load_current(state)
    assert point is not None
    assert point.manifest["command_cursor"] == 3
    assert point.manifest["active_command_seq"] is None
    resumed = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
    )
    assert resumed.returncode == 0, resumed.stderr + resumed.stdout
    final = json.loads(report.read_text(encoding="utf-8"))
    baseline_report = baseline["report"]
    assert final["output"]["sha256"] == baseline_report["output"]["sha256"]
    assert final["budget"] == baseline_report["budget"]
    assert final["segments"] == baseline_report["segments"]
    assert final["reuse"] == baseline_report["reuse"]
    assert final["recompute"] == baseline_report["recompute"]


def test_forced_kill_at_intra_command_checkpoint_and_recovery(
    baseline: dict[str, object], tmp_path: Path
) -> None:
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report = tmp_path / "report.json"
    crashed = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
        "--fresh",
        "--crash-on-checkpoint",
        "2",
    )
    assert crashed.returncode == 137
    point = load_current(state)
    assert point is not None
    assert point.manifest["command_cursor"] == 0
    assert point.manifest["active_command_seq"] == 1
    assert point.manifest["command_complete"] is False
    assert point.manifest["consumed_frame_budget"] == 91
    resumed = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report,
    )
    assert resumed.returncode == 0, resumed.stderr + resumed.stdout
    final = json.loads(report.read_text(encoding="utf-8"))
    baseline_report = baseline["report"]
    assert final["output"]["sha256"] == baseline_report["output"]["sha256"]
    assert final["budget"] == baseline_report["budget"]


def test_graceful_signal_finishes_command_and_exits_zero(tmp_path: Path) -> None:
    if os.name == "nt":
        return
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report_path = tmp_path / "report.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vc_workbench",
            "run",
            "--samples",
            str(SAMPLES),
            "--config",
            str(SAMPLES / "config.json"),
            "--commands",
            str(SAMPLES / "commands.json"),
            "--state",
            str(state),
            "--output",
            str(output),
            "--report",
            str(report_path),
            "--fresh",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    while not (state / "CURRENT").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.8)
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, stderr + stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exit_code"] == 0
    assert report["run_status"] == "checkpoint"
    assert report["active_command_seq"] is None
    assert report["command_cursor"] >= 1

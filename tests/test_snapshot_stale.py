from __future__ import annotations

import json
from pathlib import Path

from vc_workbench.state import load_current

from conftest import SAMPLES, run_cli


def test_config_change_reports_stale_and_rebuilds(samples_copy: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report_path = tmp_path / "report.json"
    first = run_cli(
        "run",
        "--samples",
        samples_copy,
        "--config",
        samples_copy / "config.json",
        "--commands",
        samples_copy / "commands.json",
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
    assert first.returncode == 0, first.stderr + first.stdout
    config = json.loads((samples_copy / "config.json").read_text(encoding="utf-8"))
    config["thread_pool_workers"] = 2
    changed_config = samples_copy / "changed_config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    second = run_cli(
        "run",
        "--samples",
        samples_copy,
        "--config",
        changed_config,
        "--commands",
        samples_copy / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report_path,
        "--stop-after-command",
        "1",
    )
    assert second.returncode == 0, second.stderr + second.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [event["code"] for event in report["startup_events"]] == ["E_SNAPSHOT_STALE"]
    assert report["recovery"]["recovered"] is False
    assert report["command_cursor"] == 1
    current = load_current(state)
    assert current is not None
    assert current.manifest["config_hash"] == report["config_hash"]


def test_corrupt_stage_artifact_reports_stale_and_rebuilds(samples_copy: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    output = tmp_path / "out.wav"
    report_path = tmp_path / "report.json"
    first = run_cli(
        "run",
        "--samples",
        samples_copy,
        "--config",
        samples_copy / "config.json",
        "--commands",
        samples_copy / "commands.json",
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
    assert first.returncode == 0
    current = load_current(state)
    assert current is not None
    snapshot_dir = state / "snapshots" / current.snapshot_id
    (snapshot_dir / "stats.json").write_bytes(b"corrupt")
    second = run_cli(
        "run",
        "--samples",
        samples_copy,
        "--config",
        samples_copy / "config.json",
        "--commands",
        samples_copy / "commands.json",
        "--state",
        state,
        "--output",
        output,
        "--report",
        report_path,
        "--stop-after-command",
        "1",
    )
    assert second.returncode == 0, second.stderr + second.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [event["code"] for event in report["startup_events"]] == ["E_SNAPSHOT_STALE"]

from __future__ import annotations

import json
from pathlib import Path

from conftest import SAMPLES, run_cli


def test_serial_and_parallel_semantics_are_identical(tmp_path: Path) -> None:
    plan = json.loads((SAMPLES / "commands.json").read_text(encoding="utf-8"))
    plan["commands"] = plan["commands"][:1]
    plan_path = tmp_path / "one_command.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    reports = []
    outputs = []
    for workers in (1, 4):
        config = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
        config["thread_pool_workers"] = workers
        config_path = tmp_path / f"config_{workers}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        state = tmp_path / f"state_{workers}"
        output = tmp_path / f"out_{workers}.wav"
        report_path = tmp_path / f"report_{workers}.json"
        result = run_cli(
            "run",
            "--samples",
            SAMPLES,
            "--config",
            config_path,
            "--commands",
            plan_path,
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
        assert result.returncode == 0, result.stderr + result.stdout
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
    for field in ("segments", "reuse", "recompute", "failures", "assembly", "budget", "speaker_stats"):
        assert reports[0][field] == reports[1][field]


def test_segment_faults_are_isolated_and_never_assembled(tmp_path: Path) -> None:
    plan = json.loads((SAMPLES / "commands.json").read_text(encoding="utf-8"))
    plan["initial_segments"] = plan["initial_segments"] + [
        {"segment_id": "seg_004", "path": "sources/append.wav", "start_sample": 0, "end_sample": None}
    ]
    plan["commands"] = [{"seq": 1, "op": "set_timbre_strength", "value": 0.71}]
    plan["fault_injection"] = [
        {"segment_id": "seg_001", "stage": "analysis", "exception_type": "InjectedAnalysis"},
        {"segment_id": "seg_002", "stage": "mapping", "exception_type": "InjectedMapping"},
        {"segment_id": "seg_003", "stage": "synthesis", "exception_type": "InjectedSynthesis"},
    ]
    plan_path = tmp_path / "fault_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    report_path = tmp_path / "report.json"
    result = run_cli(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        plan_path,
        "--state",
        tmp_path / "state",
        "--output",
        tmp_path / "out.wav",
        "--report",
        report_path,
        "--fresh",
    )
    assert result.returncode == 4
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [(failure["segment_id"], failure["stage"]) for failure in report["failures"]] == [
        ("seg_001", "analysis"),
        ("seg_002", "mapping"),
        ("seg_003", "synthesis"),
    ]
    assert [item["segment_id"] for item in report["assembly"]["included"]] == ["seg_004"]
    assert {item["segment_id"] for item in report["assembly"]["skipped"]} == {"seg_001", "seg_002", "seg_003"}


def test_business_error_codes_and_exit_status(tmp_path: Path) -> None:
    cases = [
        (
            SAMPLES / "errors" / "audio_format",
            SAMPLES / "errors" / "audio_format" / "config.json",
            SAMPLES / "errors" / "audio_format" / "commands.json",
            ["E_AUDIO_FORMAT", "E_AUDIO_FORMAT"],
        ),
        (
            SAMPLES,
            SAMPLES / "errors" / "overflow_config.json",
            SAMPLES / "errors" / "overflow_commands.json",
            ["E_SEGMENT_OVERFLOW"],
        ),
        (
            SAMPLES,
            SAMPLES / "config.json",
            SAMPLES / "errors" / "invalid_commands.json",
            ["E_COMMAND_INVALID", "E_COMMAND_INVALID"],
        ),
    ]
    for index, (samples, config, commands, expected_codes) in enumerate(cases):
        report_path = tmp_path / f"report_{index}.json"
        result = run_cli(
            "run",
            "--samples",
            samples,
            "--config",
            config,
            "--commands",
            commands,
            "--state",
            tmp_path / f"state_{index}",
            "--output",
            tmp_path / f"out_{index}.wav",
            "--report",
            report_path,
            "--fresh",
        )
        assert result.returncode == 4
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert [failure["code"] for failure in report["failures"]] == expected_codes

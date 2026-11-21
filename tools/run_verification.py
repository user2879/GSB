"""Run the local acceptance matrix and write artifacts/verification_summary.json."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_samples import generate_samples
from vc_workbench.state import load_current, load_snapshot_directory


SAMPLES = ROOT / "samples"
ARTIFACTS = ROOT / "artifacts"
WORK = ARTIFACTS / "verification"


def _run(*args: str | Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vc_workbench", *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_snapshot(state_root: Path, cursor: int):
    for snapshot_dir in sorted((state_root / "snapshots").iterdir()):
        if not snapshot_dir.is_dir():
            continue
        manifest = _read(snapshot_dir / "manifest.json")
        if (
            manifest.get("command_complete") is True
            and manifest.get("active_command_seq") is None
            and int(manifest.get("command_cursor", -1)) == cursor
        ):
            return load_snapshot_directory(snapshot_dir)
    raise RuntimeError(f"no completed snapshot for cursor {cursor}")


def _record(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _run_graceful_signal(state: Path, output: Path, report: Path) -> tuple[int, dict[str, Any]]:
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
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
            str(report),
            "--fresh",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    deadline = time.monotonic() + 10
    while not (state / "CURRENT").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.8)
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=60)
    if not report.exists():
        raise RuntimeError(f"graceful signal produced no report: {stdout}\n{stderr}")
    return process.returncode, _read(report)


def main() -> int:
    started = time.monotonic()
    checks: list[dict[str, Any]] = []
    if not (SAMPLES / "commands.json").exists():
        generate_samples(SAMPLES, force=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    normal_state = WORK / "normal_state"
    normal_output = ARTIFACTS / "sample_converted.wav"
    normal_report_path = ARTIFACTS / "sample_run_report.json"
    normal = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        normal_state,
        "--output",
        normal_output,
        "--report",
        normal_report_path,
        "--fresh",
    )
    normal_report = _read(normal_report_path)
    shutil.copyfile(normal_report_path, WORK / "normal_report.json")
    _record(
        checks,
        "normal_sample_run",
        normal.returncode == 0 and normal_report["command_cursor"] == 8 and normal_report["failures"] == [],
        {
            "exit_code": normal.returncode,
            "command_count": normal_report["command_count"],
            "assembly_sha256": normal_report["assembly"]["pcm_sha256"],
            "output_bytes": normal_output.stat().st_size,
            "budget_consumed": normal_report["budget"]["consumed"],
        },
    )

    incremental_state = WORK / "incremental_state"
    incremental_output = WORK / "incremental.wav"
    incremental_report_path = WORK / "incremental.json"
    prefixes: list[dict[str, Any]] = []
    for cursor in range(1, 9):
        result = _run(
            "run",
            "--samples",
            SAMPLES,
            "--config",
            SAMPLES / "config.json",
            "--commands",
            SAMPLES / "commands.json",
            "--state",
            incremental_state,
            "--output",
            incremental_output,
            "--report",
            incremental_report_path,
            "--stop-after-command",
            cursor,
        )
        current = load_current(incremental_state)
        full = _completed_snapshot(normal_state, cursor)
        report = _read(incremental_report_path)
        equal = (
            result.returncode == 0
            and current is not None
            and current.snapshot_id == full.snapshot_id
            and current.analysis == full.analysis
            and current.stats == full.stats
            and current.mappings == full.mappings
            and current.assembly == full.assembly
            and current.ledger == full.ledger
            and current.pcm_by_segment == full.pcm_by_segment
            and current.assembly_pcm == full.assembly_pcm
            and report["reuse"] == current.ledger["decision"]["reuse"]
            and report["recompute"] == current.ledger["decision"]["recompute"]
            and report["failures"] == current.ledger["failures"]
        )
        prefixes.append(
            {
                "prefix": cursor,
                "equal": equal,
                "snapshot_id": current.snapshot_id if current else None,
                "reused": [item["segment_id"] for item in report["reuse"]],
                "recomputed": [item["segment_id"] for item in report["recompute"]],
                "assembly_sha256": report["assembly"]["pcm_sha256"],
            }
        )
    _record(
        checks,
        "incremental_prefixes_equal_full_replay",
        all(item["equal"] for item in prefixes),
        {"prefixes": prefixes},
    )

    crash_state = WORK / "crash_after_state"
    crash_output = WORK / "crash_after.wav"
    crash_report = WORK / "crash_after.json"
    crash = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        crash_state,
        "--output",
        crash_output,
        "--report",
        crash_report,
        "--fresh",
        "--crash-after-command",
        "3",
    )
    crash_point = load_current(crash_state)
    resumed = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        crash_state,
        "--output",
        crash_output,
        "--report",
        crash_report,
    )
    crash_final = _read(crash_report)
    crash_details = {
        "crash_exit": crash.returncode,
        "recovery_point": {
            "command_cursor": crash_point.manifest["command_cursor"] if crash_point else None,
            "active_command_seq": crash_point.manifest["active_command_seq"] if crash_point else None,
            "consumed_frame_budget": crash_point.manifest["consumed_frame_budget"] if crash_point else None,
            "snapshot_id": crash_point.snapshot_id if crash_point else None,
        },
        "resume_exit": resumed.returncode,
        "assembly_sha256": crash_final["assembly"]["pcm_sha256"],
        "same_as_full": crash_final["assembly"]["pcm_sha256"] == normal_report["assembly"]["pcm_sha256"],
    }

    intra_state = WORK / "intra_command_state"
    intra_output = WORK / "intra_command.wav"
    intra_report = WORK / "intra_command.json"
    intra = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        intra_state,
        "--output",
        intra_output,
        "--report",
        intra_report,
        "--fresh",
        "--crash-on-checkpoint",
        "2",
    )
    intra_point = load_current(intra_state)
    intra_resumed = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        intra_state,
        "--output",
        intra_output,
        "--report",
        intra_report,
    )
    intra_final = _read(intra_report)
    intra_details = {
        "crash_exit": intra.returncode,
        "recovery_point": {
            "command_cursor": intra_point.manifest["command_cursor"] if intra_point else None,
            "active_command_seq": intra_point.manifest["active_command_seq"] if intra_point else None,
            "command_complete": intra_point.manifest["command_complete"] if intra_point else None,
            "consumed_frame_budget": intra_point.manifest["consumed_frame_budget"] if intra_point else None,
            "snapshot_id": intra_point.snapshot_id if intra_point else None,
        },
        "resume_exit": intra_resumed.returncode,
        "assembly_sha256": intra_final["assembly"]["pcm_sha256"],
        "same_as_full": intra_final["assembly"]["pcm_sha256"] == normal_report["assembly"]["pcm_sha256"],
        "same_budget": intra_final["budget"] == normal_report["budget"],
    }
    _record(
        checks,
        "forced_kill_recovery",
        bool(
            crash.returncode == 137
            and crash_details["same_as_full"]
            and intra.returncode == 137
            and intra_details["recovery_point"]["active_command_seq"] == 1
            and intra_details["same_as_full"]
            and intra_details["same_budget"]
        ),
        {"after_command": crash_details, "intra_command": intra_details},
    )

    try:
        graceful_code, graceful_report = _run_graceful_signal(
            WORK / "graceful_state", WORK / "graceful.wav", WORK / "graceful.json"
        )
        graceful_details = {
            "exit_code": graceful_code,
            "run_status": graceful_report["run_status"],
            "command_cursor": graceful_report["command_cursor"],
            "active_command_seq": graceful_report["active_command_seq"],
            "report_written": True,
        }
        graceful_ok = graceful_code == 0 and graceful_report["active_command_seq"] is None
    except Exception as exc:
        graceful_details = {"error": f"{type(exc).__name__}: {exc}"}
        graceful_ok = False
    _record(checks, "graceful_signal_exit", graceful_ok, graceful_details)

    budget_config = _read(SAMPLES / "config.json")
    budget_config["speech_frame_budget"] = 70
    budget_config_path = WORK / "budget_config.json"
    budget_config_path.write_text(json.dumps(budget_config, indent=2, sort_keys=True), encoding="utf-8")
    budget_report_path = WORK / "budget_report.json"
    budget_run = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        budget_config_path,
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        WORK / "budget_state",
        "--output",
        WORK / "budget.wav",
        "--report",
        budget_report_path,
        "--fresh",
        "--stop-after-command",
        "1",
    )
    budget = _read(budget_report_path)
    budget_details = {
        "exit_code": budget_run.returncode,
        "budget": budget["budget"],
        "failures": [failure["code"] for failure in budget["failures"]],
        "assembly": {
            "status": budget["assembly"]["status"],
            "sample_count": budget["assembly"]["sample_count"],
            "included": [item["segment_id"] for item in budget["assembly"]["included"]],
        },
    }
    _record(
        checks,
        "frame_budget_exhaustion_and_todo",
        budget_run.returncode == 4
        and budget["budget"]["consumed"] == 49
        and budget["budget"]["remaining"] == 21
        and budget["budget"]["todo"] == ["seg_002", "seg_003"]
        and budget["budget"]["todo_cursor"] == 1
        and budget["budget"]["replay_ok"]
        and budget["budget"]["todo_replay_ok"]
        and budget["assembly"]["status"] == "partial"
        and [item["segment_id"] for item in budget["assembly"]["included"]] == ["seg_001"],
        budget_details,
    )

    error_cases = [
        (
            "audio_format",
            SAMPLES / "errors" / "audio_format",
            SAMPLES / "errors" / "audio_format" / "config.json",
            SAMPLES / "errors" / "audio_format" / "commands.json",
            ["E_AUDIO_FORMAT", "E_AUDIO_FORMAT"],
        ),
        (
            "segment_overflow",
            SAMPLES,
            SAMPLES / "errors" / "overflow_config.json",
            SAMPLES / "errors" / "overflow_commands.json",
            ["E_SEGMENT_OVERFLOW"],
        ),
        (
            "invalid_command",
            SAMPLES,
            SAMPLES / "config.json",
            SAMPLES / "errors" / "invalid_commands.json",
            ["E_COMMAND_INVALID", "E_COMMAND_INVALID"],
        ),
    ]
    error_details: dict[str, Any] = {}
    errors_ok = True
    for name, samples, config, commands, expected in error_cases:
        report_path = WORK / f"error_{name}.json"
        result = _run(
            "run",
            "--samples",
            samples,
            "--config",
            config,
            "--commands",
            commands,
            "--state",
            WORK / f"error_{name}_state",
            "--output",
            WORK / f"error_{name}.wav",
            "--report",
            report_path,
            "--fresh",
        )
        report = _read(report_path)
        codes = [failure["code"] for failure in report["failures"]]
        ok = result.returncode == 4 and codes == expected
        errors_ok = errors_ok and ok
        error_details[name] = {"exit_code": result.returncode, "codes": codes, "passed": ok}
    _record(checks, "business_error_cases", errors_ok, error_details)

    one_plan = _read(SAMPLES / "commands.json")
    one_plan["commands"] = one_plan["commands"][:1]
    one_plan_path = WORK / "one_command_plan.json"
    one_plan_path.write_text(json.dumps(one_plan, indent=2, sort_keys=True), encoding="utf-8")
    parallel_reports: list[dict[str, Any]] = []
    parallel_outputs: list[bytes] = []
    for workers in (4, 1):
        config = _read(SAMPLES / "config.json")
        config["thread_pool_workers"] = workers
        config_path = WORK / f"workers_{workers}.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        report_path = WORK / f"workers_{workers}_report.json"
        output_path = WORK / f"workers_{workers}.wav"
        result = _run(
            "run",
            "--samples",
            SAMPLES,
            "--config",
            config_path,
            "--commands",
            one_plan_path,
            "--state",
            WORK / f"workers_{workers}_state",
            "--output",
            output_path,
            "--report",
            report_path,
            "--fresh",
            "--stop-after-command",
            "1",
        )
        assert result.returncode == 0, result.stderr + result.stdout
        parallel_reports.append(_read(report_path))
        parallel_outputs.append(output_path.read_bytes())
    parallel_fields_equal = all(
        parallel_reports[0][field] == parallel_reports[1][field]
        for field in ("segments", "reuse", "recompute", "failures", "assembly", "budget", "speaker_stats")
    )
    _record(
        checks,
        "serial_vs_parallel_semantics",
        parallel_outputs[0] == parallel_outputs[1] and parallel_fields_equal,
        {
            "configured_workers": [4, 1],
            "parallel_assembly_sha256": parallel_reports[0]["assembly"]["pcm_sha256"],
            "serial_assembly_sha256": parallel_reports[1]["assembly"]["pcm_sha256"],
            "fields_equal": parallel_fields_equal,
        },
    )

    stale_state = WORK / "stale_state"
    stale_report = WORK / "stale_report.json"
    first_stale = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        stale_state,
        "--output",
        WORK / "stale.wav",
        "--report",
        stale_report,
        "--fresh",
        "--stop-after-command",
        "1",
    )
    assert first_stale.returncode == 0
    changed = _read(SAMPLES / "config.json")
    changed["thread_pool_workers"] = 2
    changed_path = WORK / "changed_config.json"
    changed_path.write_text(json.dumps(changed, indent=2, sort_keys=True), encoding="utf-8")
    second_stale = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        changed_path,
        "--commands",
        SAMPLES / "commands.json",
        "--state",
        stale_state,
        "--output",
        WORK / "stale.wav",
        "--report",
        stale_report,
        "--stop-after-command",
        "1",
    )
    stale_data = _read(stale_report)
    _record(
        checks,
        "snapshot_stale_detection",
        second_stale.returncode == 0
        and [event["code"] for event in stale_data["startup_events"]] == ["E_SNAPSHOT_STALE"]
        and stale_data["recovery"]["recovered"] is False,
        {
            "startup_events": stale_data["startup_events"],
            "rebuilt_command_cursor": stale_data["command_cursor"],
            "config_hash": stale_data["config_hash"],
        },
    )

    fault_plan = _read(SAMPLES / "commands.json")
    fault_plan["initial_segments"] = fault_plan["initial_segments"] + [
        {"segment_id": "seg_004", "path": "sources/append.wav", "start_sample": 0, "end_sample": None}
    ]
    fault_plan["commands"] = [{"seq": 1, "op": "set_timbre_strength", "value": 0.71}]
    fault_plan["fault_injection"] = [
        {"segment_id": "seg_001", "stage": "analysis", "exception_type": "InjectedAnalysis"},
        {"segment_id": "seg_002", "stage": "mapping", "exception_type": "InjectedMapping"},
        {"segment_id": "seg_003", "stage": "synthesis", "exception_type": "InjectedSynthesis"},
    ]
    fault_path = WORK / "fault_plan.json"
    fault_path.write_text(json.dumps(fault_plan, indent=2, sort_keys=True), encoding="utf-8")
    fault_report_path = WORK / "fault_report.json"
    fault_run = _run(
        "run",
        "--samples",
        SAMPLES,
        "--config",
        SAMPLES / "config.json",
        "--commands",
        fault_path,
        "--state",
        WORK / "fault_state",
        "--output",
        WORK / "fault.wav",
        "--report",
        fault_report_path,
        "--fresh",
    )
    fault = _read(fault_report_path)
    fault_stages = [(failure["segment_id"], failure["stage"]) for failure in fault["failures"]]
    _record(
        checks,
        "segment_failure_isolation",
        fault_run.returncode == 4
        and fault_stages
        == [("seg_001", "analysis"), ("seg_002", "mapping"), ("seg_003", "synthesis")]
        and [item["segment_id"] for item in fault["assembly"]["included"]] == ["seg_004"],
        {
            "failures": fault_stages,
            "assembly_included": [item["segment_id"] for item in fault["assembly"]["included"]],
            "assembly_skipped": [item["segment_id"] for item in fault["assembly"]["skipped"]],
        },
    )

    generated_a = WORK / "generated_a"
    generated_b = WORK / "generated_b"
    generate_samples(generated_a, force=True)
    generate_samples(generated_b, force=True)
    files_a = {
        path.relative_to(generated_a).as_posix(): _sha(path)
        for path in sorted(generated_a.rglob("*"))
        if path.is_file() and path.name != ".generated"
    }
    files_b = {
        path.relative_to(generated_b).as_posix(): _sha(path)
        for path in sorted(generated_b.rglob("*"))
        if path.is_file() and path.name != ".generated"
    }
    _record(
        checks,
        "fixed_seed_sample_determinism",
        files_a == files_b,
        {"file_count": len(files_a), "seed": 20261121},
    )

    # Keep reports and hashes, not multi-megabyte intermediate snapshot trees.
    for directory in sorted(WORK.iterdir()):
        if directory.is_dir() and (directory.name.endswith("_state") or directory.name.startswith("generated_")):
            shutil.rmtree(directory, ignore_errors=True)
    summary = {
        "verification_format": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os_name": os.name,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
    (ARTIFACTS / "verification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("all_passed", "elapsed_seconds", "checks")}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

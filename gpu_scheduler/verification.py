from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .app import export_receipts, recover_only, run_scheduler, verify_receipt_export
from .errors import AppError, DependencyCycleError, PortInUseError, SnapshotDigestError, SpecError
from .hashing import canonical_text
from .ledger import Ledger
from .manager import ExecutorManager, occupy_port_until_released
from .quant import build_watermarks
from .snapshot import restore_and_verify
from .spec import load_spec
from .viewmodel import compare_cache_file

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def _check(name: str, condition: bool, detail: Any = None) -> dict[str, Any]:
    if not condition:
        raise AssertionError(f"acceptance check failed: {name}: {detail}")
    return {"name": name, "ok": True, "detail": detail}


def watermark_checks(run_dir: Path) -> dict[str, Any]:
    spec = load_spec(SAMPLES / "preemption.json")
    watermarks = build_watermarks(spec, {"alpha": 4}, {"alpha": 16384}, {"alpha": 2})
    gpu = watermarks["global"]["gpu_quota"]
    pool = watermarks["pools"][0]["gpu_physical"]
    checks = [
        _check("global_scale_factor_is_1000", gpu["scale_factor"] == 1000, gpu["scale_factor"]),
        _check("global_full_watermark", (gpu["used"], gpu["capacity"], gpu["remaining"], gpu["ratio_fixed"]) == (4, 4, 0, 1000)),
        _check("truncation_is_floor_down", gpu["truncation_direction"] == "floor_down_for_nonnegative_values"),
        _check("error_bound_is_one_fixed_unit", gpu["quant_error_bound"]["absolute_ratio_upper_bound_exclusive"] == "1/1000"),
        _check("pool_ratio_matches", pool["ratio_fixed"] == 1000),
        _check("pool_order_stable", [p["id"] for p in watermarks["pools"]] == sorted(p["id"] for p in watermarks["pools"])),
    ]
    return {"ok": True, "checks": checks, "watermarks": watermarks}


def preemption_checks(run_dir: Path) -> dict[str, Any]:
    directory = run_dir / "preemption"
    spec = load_spec(SAMPLES / "preemption.json")
    report = run_scheduler(SAMPLES / "preemption.json", directory, max_wall_seconds=40)
    ledger = Ledger(directory / "ledger.sqlite3")
    try:
        preemptions = [dict(r) for r in ledger.conn.execute("SELECT * FROM preemption_rows ORDER BY id")]
        low_new_zero = ledger.conn.execute(
            "SELECT COUNT(*) AS n FROM receipts WHERE shard_id='low-new#0'"
        ).fetchone()["n"]
        cursor = ledger.conn.execute("SELECT * FROM cursor_rows WHERE task_id='low-new'").fetchone()
        suspension = ledger.conn.execute("SELECT * FROM suspension_rows WHERE task_id='low-new'").fetchone()
        propagation_payloads = [
            json.loads(r["payload_json"])
            for r in ledger.conn.execute("SELECT payload_json FROM propagation_rows ORDER BY id")
        ]
    finally:
        ledger.close()
    snapshot = restore_and_verify(directory / "session-snapshot.json", directory, spec)
    comparison = compare_cache_file(spec, directory)
    checks = [
        _check("all_eight_shards_completed", report["counts"].get("completed") == 8, report["counts"]),
        _check("preemption_recorded", len(preemptions) >= 1, len(preemptions)),
        _check("stable_victim_is_low_new", preemptions and preemptions[0]["victim_task_id"] == "low-new", preemptions[0] if preemptions else None),
        _check("victim_set_is_singleton_low_new", preemptions and json.loads(preemptions[0]["victim_set_json"]) == ["low-new"]),
        _check("preempted_shard_not_rerun", low_new_zero == 1, low_new_zero),
        _check("final_cursor_reaches_end", json.loads(cursor["pending_shards_json"]) == [], dict(cursor)),
        _check("suspension_cursor_resumed_from_shard_1", suspension is not None and json.loads(suspension["cursor_json"])["next_shard_id"] == "low-new#1", dict(suspension)),
        _check("suspension_has_renewal_or_resume", suspension is not None and suspension["renew_count"] >= 1, dict(suspension)),
        _check("preemption_propagation_recorded", any(p.get("trigger_type") == "preempted" for p in propagation_payloads), len(propagation_payloads)),
        _check("snapshot_replay_matches", snapshot["comparison"]["match"] is True, snapshot),
        _check("ui_cache_bytes_match_authoritative_view", comparison["byte_equal"] is True, comparison.get("first_difference")),
    ]
    return {"ok": True, "checks": checks, "report": report, "preemption_count": len(preemptions)}


def propagation_checks(run_dir: Path) -> dict[str, Any]:
    directory = run_dir / "propagation"
    spec = load_spec(SAMPLES / "propagation.json")
    failed = run_scheduler(
        SAMPLES / "propagation.json",
        directory,
        fail_shards={"etl#0"},
        max_wall_seconds=30,
    )
    checks = [
        _check("injected_failure_counts", failed["counts"].get("failed") == 1 and failed["counts"].get("blocked") == 3, failed["counts"]),
    ]
    before = directory / "receipts-before.json"
    export_receipts(directory, before)
    ledger = Ledger(directory / "ledger.sqlite3")
    try:
        row = ledger.conn.execute("SELECT affected_json,trace_json FROM propagation_rows ORDER BY id LIMIT 1").fetchone()
        affected = set(json.loads(row["affected_json"]))
        trace = json.loads(row["trace_json"])
    finally:
        ledger.close()
    (directory / "affected.json").write_text(canonical_text({"affected": sorted(affected)}) + "\n", encoding="utf-8")

    retried = run_scheduler(
        SAMPLES / "propagation.json",
        directory,
        retry_shards={"etl#0"},
        max_wall_seconds=30,
    )
    stability = verify_receipt_export(directory, before, affected)
    snapshot = restore_and_verify(directory / "session-snapshot.json", directory, spec)
    comparison = compare_cache_file(spec, directory)
    checks.extend(
        [
            _check("affected_set_is_four_shards", affected == {"etl#0", "etl#1", "etl#2", "mart#0"}, sorted(affected)),
            _check("stable_trace_starts_with_source", trace[0]["shard_id"] == "etl#0" and len(trace) == 4),
            _check("affected_set_rerun_to_completion", retried["counts"].get("completed") == 6, retried["counts"]),
            _check("unaffected_receipts_byte_identical", stability["ok"] and stability["byte_identical_shards"] == ["branch#0", "branch#1"], stability),
            _check("propagation_snapshot_replay", snapshot["comparison"]["match"] is True),
            _check("propagation_ui_bytes", comparison["byte_equal"] is True, comparison.get("first_difference")),
        ]
    )
    return {"ok": True, "checks": checks, "affected": sorted(affected), "stability": stability}


def _wait_for_port_closed(port: int, timeout: float = 6.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = socket.socket()
        probe.settimeout(0.2)
        result = probe.connect_ex(("127.0.0.1", port))
        probe.close()
        if result != 0:
            return
        time.sleep(0.1)


def _spawn_cli(workdir: Path, spec: Path, crash_file: Path) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    stdout = open(workdir / "cli.stdout.log", "ab")
    stderr = open(workdir / "cli.stderr.log", "ab")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpu_scheduler",
            "--workdir",
            str(workdir),
            "run",
            str(spec),
            "--crash-ready-file",
            str(crash_file),
            "--max-wall-seconds",
            "30",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=stdout,
        stderr=stderr,
    )


def kill_and_signal_checks(run_dir: Path) -> dict[str, Any]:
    kill_dir = run_dir / "kill"
    kill_dir.mkdir(parents=True, exist_ok=True)
    crash_file = kill_dir / "crash-ready.json"
    proc = _spawn_cli(kill_dir, SAMPLES / "recovery.json", crash_file)
    deadline = time.time() + 15
    while not crash_file.exists() and time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)
    crash_info = json.loads(crash_file.read_text(encoding="utf-8"))
    proc.kill()
    returncode = proc.wait(timeout=10)
    spec = load_spec(SAMPLES / "recovery.json")
    _wait_for_port_closed(spec.pools[0].port)
    recovery = recover_only(SAMPLES / "recovery.json", kill_dir)
    rerun = run_scheduler(SAMPLES / "recovery.json", kill_dir, max_wall_seconds=30)
    exit_reports = list((kill_dir / "reports").glob("exit-report-*.json")) if (kill_dir / "reports").exists() else []
    kill_checks = [
        _check("crash_ready_had_active_shards", len(crash_info.get("active_shards", [])) >= 1, crash_info),
        _check("sigkill_returncode", returncode == -9, returncode),
        _check("recovery_marks_unknown", len(recovery["unknown_shards"]) >= 1, recovery),
        _check("unknown_shards_rerun_completed", rerun["counts"].get("completed") == 2, rerun["counts"]),
        _check("sigkill_writes_no_exit_report", exit_reports == [], [str(p) for p in exit_reports]),
    ]

    signal_dir = run_dir / "signal"
    signal_dir.mkdir(parents=True, exist_ok=True)
    signal_crash = signal_dir / "crash-ready.json"
    signal_proc = _spawn_cli(signal_dir, SAMPLES / "preemption.json", signal_crash)
    deadline = time.time() + 15
    while not signal_crash.exists() and time.time() < deadline and signal_proc.poll() is None:
        time.sleep(0.05)
    signal_proc.send_signal(__import__("signal").SIGTERM)
    signal_rc = signal_proc.wait(timeout=20)
    report_path = signal_dir / "reports" / "latest-exit-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    signal_spec = load_spec(SAMPLES / "preemption.json")
    _wait_for_port_closed(signal_spec.pools[0].port)
    signal_checks = [
        _check("sigterm_returns_distinct_signal_code", signal_rc == 143, signal_rc),
        _check("sigterm_writes_complete_report", report.get("complete") is True and report.get("signal") == "SIGTERM", str(report_path)),
        _check("sigterm_report_has_ledger_cursor", isinstance(report.get("ledger_cursor"), int) and report["ledger_cursor"] >= 1),
    ]
    return {"ok": True, "kill_checks": kill_checks, "signal_checks": signal_checks, "recovery": recovery}


def error_code_checks(run_dir: Path) -> dict[str, Any]:
    error_dir = run_dir / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    bad_spec = error_dir / "bad-e-spec.json"
    raw = json.loads((SAMPLES / "preemption.json").read_text(encoding="utf-8"))
    raw["pools"][0]["host"] = "localhost"
    bad_spec.write_text(canonical_text(raw) + "\n", encoding="utf-8")

    results: dict[str, str] = {}

    def expect_code(name: str, code: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except AppError as exc:
            results[name] = exc.code
            if exc.code != code:
                raise AssertionError(f"{name}: expected {code}, got {exc.code}") from exc
            return
        raise AssertionError(f"{name}: expected {code}, no error")

    expect_code("E_SPEC", "E_SPEC", lambda: load_spec(bad_spec))
    expect_code("E_CYCLE", "E_CYCLE", lambda: load_spec(SAMPLES / "cycle_injection.json"))

    spec = load_spec(SAMPLES / "preemption.json")
    sock = occupy_port_until_released(spec.pools[0].port)
    manager = ExecutorManager(spec, error_dir / "port")
    try:
        expect_code("E_PORT", "E_PORT", manager.start)
    finally:
        sock.close()
        manager.stop()

    snapshot_dir = error_dir / "snapshot"
    run_scheduler(SAMPLES / "propagation.json", snapshot_dir, max_wall_seconds=30)
    ledger = Ledger(snapshot_dir / "ledger.sqlite3")
    try:
        ledger.tamper_latest_content_digest()
    finally:
        ledger.close()
    expect_code(
        "E_SNAPSHOT",
        "E_SNAPSHOT",
        lambda: restore_and_verify(snapshot_dir / "session-snapshot.json", snapshot_dir, load_spec(SAMPLES / "propagation.json")),
    )
    checks = [_check(name, code == expected, {"actual": code, "expected": expected}) for (name, code), expected in zip(
        sorted(results.items()), ["E_CYCLE", "E_PORT", "E_SNAPSHOT", "E_SPEC"]
    )]
    return {"ok": True, "checks": checks, "codes": results}


def run_acceptance(workdir: Path, only: str | None = None) -> dict[str, Any]:
    workdir = Path(workdir)
    run_dir = workdir / f"accept-{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, Any] = {}
    runners = {
        "watermark": watermark_checks,
        "preemption": preemption_checks,
        "propagation": propagation_checks,
        "kill": kill_and_signal_checks,
        "errors": error_code_checks,
    }
    selected = list(runners) if not only or only == "all" else [only]
    failures: list[dict[str, str]] = []
    for name in selected:
        try:
            sections[name] = runners[name](run_dir)
        except Exception as exc:
            sections[name] = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            failures.append({"section": name, "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "ok": not failures,
        "run_dir": str(run_dir),
        "failures": failures,
        "sections": {name: {"ok": section.get("ok"), "check_count": len(section.get("checks", section.get("kill_checks", [])))} for name, section in sections.items()},
        "full_report_path": str(run_dir / "acceptance-report.json"),
    }
    full_report = {"summary": result, "sections": sections}
    (run_dir / "acceptance-report.json").write_text(canonical_text(full_report) + "\n", encoding="utf-8")
    latest = workdir / "latest-report.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(canonical_text(result) + "\n", encoding="utf-8")
    return result

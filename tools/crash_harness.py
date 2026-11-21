"""Process-level forced-termination, recovery and fault-injection harness.

The harness never fabricates a shutdown report for a killed process.  It starts
the real CLI server on loopback, kills it with the platform's forceful process
kill, then asks the recovery tool to reconstruct state from SQLite.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from txn_mw.canonical import canonical_json
from txn_mw.cli import inject_tail_truncation
from txn_mw.scenario import QueryStubServer
from txn_mw.server import request
from txn_mw.verify import verify_ledger_file


def _wait_json(path: Path, timeout_s: float = 12.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {path}")


def _start_server(db: Path, port_file: Path, stdout_path: Path, *extra: str) -> subprocess.Popen[str]:
    if port_file.exists():
        port_file.unlink()
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "serve",
        "--db",
        str(db),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--port-file",
        str(port_file),
        "--query-budget-ms",
        "5000",
        "--pull-budget-ms",
        "50",
        "--query-retry-ms",
        "20",
        "--query-tick-ms",
        "5",
        "--log",
        str(port_file.parent / "structured.jsonl"),
        *extra,
    ]
    output = stdout_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, text=True)
    process._output_handle = output  # type: ignore[attr-defined]
    return process


def _wait_trace(address: tuple[str, int], txn_id: str, predicate, timeout_s: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            last = request(*address, {"op": "trace", "txn_id": txn_id})
            if predicate(last):
                return last
        except (ConnectionError, OSError):
            pass
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {txn_id}; last={canonical_json(last)}")


def _kill(process: subprocess.Popen[str]) -> None:
    process.kill()
    process.wait(timeout=10)
    handle = getattr(process, "_output_handle", None)
    if handle:
        handle.close()


def _terminate_gracefully(process: subprocess.Popen[str]) -> int:
    if process.poll() is None:
        try:
            process.send_signal(signal.SIGTERM)
        except (AttributeError, OSError):
            process.terminate()
    code = process.wait(timeout=10)
    handle = getattr(process, "_output_handle", None)
    if handle:
        handle.close()
    return code


def _half(
    txn_id: str,
    body: dict[str, Any],
    timeout_ms: int,
    port: int,
    query_max: int = 4,
) -> dict[str, Any]:
    return {
        "op": "half",
        "txn_id": txn_id,
        "version": 1,
        "body": body,
        "txn_timeout_ms": timeout_ms,
        "query": {"host": "127.0.0.1", "port": port},
        "query_max": query_max,
    }


def run_crash_recovery(artifacts: Path) -> dict[str, Any]:
    db = artifacts / "crash.db"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db) + suffix)
        if candidate.exists():
            candidate.unlink()
    port_file = artifacts / "crash-port.json"
    output = artifacts / "crash-server-stdout.log"
    exit_report = artifacts / "must-not-exist-forced-exit-report.json"
    if exit_report.exists():
        exit_report.unlink()
    stub = QueryStubServer(
        {
            "scripts": {
                "crash-complete-query": [{"result": "committed", "delay_ms": 5}],
                "crash-incomplete-query": [{"result": "committed", "delay_ms": 4000}],
                "crash-exhaust": [{"result": "unknown", "delay_ms": 0}],
            }
        }
    )
    process = _start_server(db, port_file, output, "--exit-report", str(exit_report))
    report: dict[str, Any] = {"ok": False, "stub_port": stub.port}
    try:
        info = _wait_json(port_file)
        address = (info["host"], int(info["port"]))
        report["first_server"] = info
        request(*address, _half("crash-complete-query", {"case": "completed-before-kill"}, 50, stub.port))
        _wait_trace(address, "crash-complete-query", lambda t: t.get("state") == "committable")
        request(*address, _half("crash-incomplete-query", {"case": "incomplete-at-kill"}, 50, stub.port))
        _wait_trace(address, "crash-incomplete-query", lambda t: bool(t.get("query_inflight")), timeout_s=5)
        request(*address, _half("crash-exhaust", {"case": "query-limit"}, 30, stub.port, query_max=1))
        _wait_trace(address, "crash-exhaust", lambda t: t.get("disposition") == "manual_review", timeout_s=5)
        _kill(process)
        report["forced_kill"] = {
            "process_exit_code": process.returncode,
            "forced_exit_report_created": exit_report.exists(),
        }
        if exit_report.exists():
            raise AssertionError("a forced process termination incorrectly produced an exit report")

        recovery_out = artifacts / "recovery-report.json"
        recovery = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main.py"),
                "recover-report",
                "--db",
                str(db),
                "--out",
                str(recovery_out),
                "--query-budget-ms",
                "5000",
                "--query-retry-ms",
                "20",
                "--query-tick-ms",
                "5",
                "--log",
                str(artifacts / "structured.jsonl"),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=30,
        )
        if recovery.returncode != 0:
            raise RuntimeError(f"recover-report failed: {recovery.stderr}\n{recovery.stdout}")
        recovery_report = json.loads(recovery.stdout.strip().splitlines()[-1])
        report["recovery"] = recovery_report
        if "crash-incomplete-query" not in recovery_report.get("incomplete_queries_marked_unknown_and_requeued", []):
            raise AssertionError("incomplete query was not marked unknown and requeued")
        if recovery_report.get("shutdown_report_available") is not False:
            raise AssertionError("recovery report incorrectly claimed a forced-termination exit report")

        stub.answers["scripts"]["crash-incomplete-query"] = [{"result": "committed", "delay_ms": 5}]
        process2 = _start_server(
            db,
            port_file,
            output,
            "--exit-report",
            str(artifacts / "graceful-exit-report.json"),
        )
        try:
            info2 = _wait_json(port_file)
            address2 = (info2["host"], int(info2["port"]))
            report["second_server"] = info2
            trace = _wait_trace(address2, "crash-incomplete-query", lambda t: t.get("state") == "committable", 10)
            report["recovered_trace"] = trace
            pull = request(
                *address2,
                {
                    "op": "pull",
                    "txn_id": "crash-incomplete-query",
                    "consumer_id": "recovery-consumer",
                    "request_id": "recovery-pull-1",
                },
            )
            report["recovery_pull"] = pull
            stats = request(*address2, {"op": "stats"})
            report["latency_after_recovery"] = stats
            if stats["overall"]["query"]["sample_count"] < 2:
                raise AssertionError("query latency samples were not preserved across forced recovery")
            if stats["overall"]["pull"]["sample_count"] < 1:
                raise AssertionError("no pull latency was recorded after recovery")
        finally:
            report["second_server_exit_code"] = _terminate_gracefully(process2)
            report["graceful_exit_report_created"] = (artifacts / "graceful-exit-report.json").exists()

        ledger = artifacts / "crash-ledger.json"
        subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "export-ledger", "--db", str(db), "--out", str(ledger)],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        report["ledger_verify"] = verify_ledger_file(ledger)

        tail_db = artifacts / "tail-truncated.db"
        source = sqlite3.connect(str(db))
        target = sqlite3.connect(str(tail_db))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        if not inject_tail_truncation(tail_db):
            raise AssertionError("tail truncation was not applied")
        tail_out = artifacts / "tail-recovery-report.json"
        tail_run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main.py"),
                "recover-report",
                "--db",
                str(tail_db),
                "--out",
                str(tail_out),
                "--query-budget-ms",
                "5000",
                "--log",
                str(artifacts / "structured.jsonl"),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=30,
        )
        tail_report = json.loads(tail_run.stdout.strip().splitlines()[-1])
        report["tail_recovery"] = tail_report
        if not tail_report.get("tail_incomplete_records"):
            raise AssertionError("tail truncation was not reported as an incomplete tail record")

        fault_db = artifacts / "startup-faults.db"
        fault_port_file = artifacts / "fault-port.json"
        fault_process = _start_server(
            fault_db,
            fault_port_file,
            artifacts / "fault-server-stdout.log",
            "--inject-fault",
            "duplicate_txn",
            "--inject-fault",
            "query_command_conflict",
        )
        try:
            fault_info = _wait_json(fault_port_file)
            report["startup_fault_info"] = fault_info
        finally:
            _terminate_gracefully(fault_process)
        log_lines = [
            json.loads(line)
            for line in (artifacts / "structured.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report["error_codes_observed"] = sorted({line["code"] for line in log_lines if line.get("level") == "error"})
        for required in ("E_LOG", "E_DUP", "E_CONFLICT", "E_QUERY"):
            if required not in report["error_codes_observed"]:
                report.setdefault("missing_codes", []).append(required)
        if report.get("missing_codes"):
            raise AssertionError(f"missing structured error codes: {report['missing_codes']}")
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if process.poll() is None:
            _kill(process)
    finally:
        stub.close()
    (artifacts / "crash-harness-result.json").write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def main() -> int:
    artifacts = ROOT / "artifacts" / "crash"
    artifacts.mkdir(parents=True, exist_ok=True)
    result = run_crash_recovery(artifacts)
    print(canonical_json(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

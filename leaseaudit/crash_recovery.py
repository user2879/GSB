from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .clock import RealMonotonicClock
from .engine import LeaseEngine
from .errors import BusinessError, ErrorCode
from .ledger import AuditLedger
from .logging_utils import StructuredLogger
from .replay import replay_ledger
from .scenario import load_scenario
from .util import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def _scenario_document() -> dict:
    return {
        "format_version": 1,
        "scenario_version": "crash-recovery-v1",
        "lease_params_version": 1,
        "cluster_capacity": 6,
        "lease_duration_ms": 60000,
        "renew_interval_ms": 1000,
        "reclaim_threshold": 250000,
        "reclaim_window_rounds": 2,
        "observation_window_ms": 1000,
        "watermark_scale": 1000000,
        "load_duration_ms": 5000,
        "sources": [
            {
                "source_id": "alpha",
                "address": "127.0.0.1",
                "port": 55801,
                "lease_share": 3,
                "reserve_quota": 1,
                "mode": "success",
                "processing_ms": 0,
                "request_interval_ms": 200,
            },
            {
                "source_id": "beta",
                "address": "127.0.0.1",
                "port": 55802,
                "lease_share": 2,
                "reserve_quota": 0,
                "mode": "success",
                "processing_ms": 0,
                "request_interval_ms": 200,
            },
        ],
    }


def run_crash_child(scenario_path: str | Path, ledger_path: str | Path) -> int:
    scenario = load_scenario(scenario_path)
    ledger = AuditLedger(ledger_path, scenario=scenario, verify=True)
    clock = RealMonotonicClock()

    def force_stop(_intent: dict) -> None:
        os._exit(91)

    engine = LeaseEngine(scenario, ledger, clock=clock, crash_after_intent=force_stop)
    try:
        engine.issue("alpha", 1, now_ms=clock.now_ms())
    finally:
        # os._exit normally prevents this path; it remains safe for non-crash failures.
        engine.close()
    return 0


def run_crash_recovery_demo(workdir: str | Path) -> dict:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(workdir / "structured.jsonl", echo=False)
    scenario_path = workdir / "crash-scenario.json"
    write_json_atomic(scenario_path, _scenario_document())
    scenario = load_scenario(scenario_path)
    ledger_path = workdir / "crash-ledger.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(ledger_path) + suffix)
        if path.exists():
            path.unlink()

    ledger = AuditLedger(ledger_path, scenario=scenario, verify=True)
    clock = RealMonotonicClock()
    engine = LeaseEngine(scenario, ledger, clock=clock, logger=logger)
    try:
        committed = engine.issue("alpha", 1, now_ms=clock.now_ms())
        committed_state = engine.snapshot()
    finally:
        engine.close()

    command = [
        sys.executable,
        "-m",
        "leaseaudit",
        "_crash-child",
        "--scenario",
        str(scenario_path),
        "--ledger",
        str(ledger_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 91:
        raise BusinessError(
            ErrorCode.E_ENV,
            "forced-termination child did not exit with the injected crash code",
            details={
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            },
        )

    recovered_ledger = AuditLedger(ledger_path, scenario=scenario, verify=True)
    recovered_engine = LeaseEngine(scenario, recovered_ledger, clock=RealMonotonicClock(), logger=logger)
    try:
        recovered_engine.process_expirations(now_ms=RealMonotonicClock().now_ms())
        recovered_state = recovered_engine.snapshot()
        integrity = recovered_ledger.verify_integrity()
        unknown = recovered_ledger.unknown_leases()
        event_types = [event.event_type for event in recovered_ledger.events()]
    finally:
        recovered_engine.close()
    audit = replay_ledger(ledger_path, scenario)
    credible = (
        completed.returncode == 91
        and len(unknown) == 1
        and recovered_state["exit_report_available"] is False
        and audit["status"] == "OK"
    )
    result = {
        "status": "OK" if credible else "E_RECOVERY",
        "business_code": "OK" if credible else "E_RECOVERY",
        "forced_exit_code": completed.returncode,
        "committed_lease": committed.as_dict(),
        "recovered_active_leases": [
            lease
            for source in recovered_state["sources"]
            for lease in source["active_leases"]
        ],
        "recovered_global_allocatable": recovered_state["global_allocatable"],
        "unknown_leases": unknown,
        "unknown_count": len(unknown),
        "lease_unknown_events": event_types.count("LEASE_UNKNOWN"),
        "exit_report_available": recovered_state["exit_report_available"],
        "reissue_required": bool(unknown),
        "audit": audit,
        "ledger": {
            "format_version": integrity["format_version"],
            "content_digest": integrity["content_digest"],
            "event_count": integrity["event_count"],
            "path": str(ledger_path),
        },
        "note": "Recovery uses committed events only; the uncommitted issuance intent is marked unknown and no forced-exit report is claimed.",
    }
    write_json_atomic(workdir / "crash-recovery-summary.json", result)
    return result

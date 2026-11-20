from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy

from gateway_quota.errors import BusinessError, VersionMismatchError
from gateway_quota.loader import load_events, load_parameter_snapshot, load_reconfigurations
from gateway_quota.logging import StructuredLogger
from gateway_quota.loopback import LoopbackSignalBroker, check_loopback_port
from gateway_quota.runner import ExperimentRunner, RunnerConfig
from gateway_quota.storage import SQLiteLedger


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
ARTIFACTS = ROOT / "artifacts"


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _event_audit_map(store: SQLiteLedger) -> dict[int, dict[str, Any]]:
    return {entry["seq"]: entry for entry in store.all_event_audits()}


def _run_crashing_child(ledger: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "gateway_quota.cli",
        "run",
        "--pools",
        str(SAMPLES / "pools.json"),
        "--borrows",
        str(SAMPLES / "borrows.json"),
        "--trace",
        str(SAMPLES / "trace.ndjson"),
        "--reconfig",
        str(SAMPLES / "reconfig_v2.json"),
        "--ledger",
        str(ledger),
        "--batch-size",
        "5",
        "--crash-after-seq",
        "13",
        "--log",
        str(log_path),
    ]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def _summarize_scenarios(store: SQLiteLedger, outcome: dict[str, Any]) -> dict[str, Any]:
    audits = _event_audit_map(store)
    transitions = store.transitions()
    if not transitions:
        raise RuntimeError("expected a committed parameter transition")
    transition = transitions[0]["transition"]
    latest_loans = {
        loan["loan_id"]: loan for loan in outcome["final_state"]["loans"]
    }
    final_pools = {pool["pool_id"]: pool for pool in outcome["final_state"]["pools"]}
    seq3 = audits[3]["result"]
    seq7 = audits[7]["result"]
    seq13 = audits[13]["result"]
    seq14 = audits[14]["result"]
    seq15 = audits[15]["result"]
    seq22 = audits[22]["result"]
    warning_alpha_at_15 = next(
        warning for warning in seq15["warnings"] if warning["pool_id"] == "alpha"
    )
    return {
        "single_pool_full": {
            "event_seq": 3,
            "alpha_capacity_in_use": seq3["audit"]["pools"]["alpha"]["capacity_in_use"],
            "alpha_remaining_connections": seq3["audit"]["pools"]["alpha"]["remaining_connections"],
            "alpha_warning": seq3["audit"]["pools"]["alpha"]["warning"],
            "rejected_establishment_event_seq": 7,
            "rejected_issue_codes": [issue["code"] for issue in seq7["issues"]],
        },
        "cross_pool_borrow_and_return": {
            "borrow_event_seq": 13,
            "loan_id": seq13["audit"]["ownership"]["b3"]["loan_id"],
            "origin_pool": seq13["audit"]["ownership"]["b3"]["origin_pool"],
            "current_owner_pool": seq13["audit"]["ownership"]["b3"]["current_owner_pool"],
            "overdue_event_seq": 14,
            "overdue_issue_codes": [issue["code"] for issue in seq14["issues"]],
            "return_event_seq": 15,
            "loan_status": latest_loans["loan-000001"]["status"],
            "borrow_ledger": latest_loans["loan-000001"],
        },
        "diagnostic_at_seq_15": {
            "remaining_connections": warning_alpha_at_15["remaining_connections"],
            "rate_estimable": warning_alpha_at_15["arrival_rate"]["estimable"],
            "arrivals_in_window": warning_alpha_at_15["arrival_rate"]["arrivals_in_window"],
            "eta_estimable": warning_alpha_at_15["exhaustion_eta"]["estimable"],
            "eta_ms": warning_alpha_at_15["exhaustion_eta"]["eta_ms"],
            "estimation_window_ms": warning_alpha_at_15["exhaustion_eta"]["estimation_window_ms"],
            "error_bound_ms": warning_alpha_at_15["exhaustion_eta"]["error_bound_ms"],
            "formula": warning_alpha_at_15["exhaustion_eta"]["formula"],
        },
        "parameter_tightening": {
            "after_event_seq": transition["after_event_seq"],
            "old_version": transition["previous_version"],
            "new_version": transition["version"],
            "affected_pools": transition["affected_pools"],
            "cleared_alert_records": transition["cleared_alert_records"],
            "invalidated_borrows": transition["invalidated_borrows"],
            "invalidated_borrow_relations": transition["invalidated_borrow_relations"],
            "added_borrow_relations": transition["added_borrow_relations"],
            "replaced_view": transition["summary"]["replaced_view"],
            "seq22_old_warning_pools": sorted(
                warning["pool_id"] for warning in seq22["warnings"]
            ),
            "post_transition_current_warnings": outcome["final_state"]["current_warnings"],
        },
        "final_pools": {
            pool_id: {
                "own_active": pool["own_active"],
                "active_connections": pool["active_connections"],
                "lent_out": pool["lent_out"],
                "capacity_in_use": pool["capacity_in_use"],
                "remaining_connections": pool["remaining_connections"],
                "warning": pool["warning"],
            }
            for pool_id, pool in sorted(final_pools.items())
        },
    }


def _validate_or_capture(action: Any) -> dict[str, Any]:
    try:
        action()
    except BusinessError as exc:
        return {
            "rejected": True,
            "envelope_code": exc.code,
            "records": [record.to_dict() for record in exc.records],
            "business_codes": sorted(
                {record.business_code for record in exc.records if record.business_code}
            ),
        }
    return {"rejected": False}


def _run_injections(selected: list[str], logger: StructuredLogger, demo_ledger: Path) -> dict[str, Any]:
    injection_dir = ARTIFACTS / "injections"
    if injection_dir.exists():
        shutil.rmtree(injection_dir)
    injection_dir.mkdir(parents=True, exist_ok=True)
    snapshot = load_parameter_snapshot(SAMPLES / "pools.json", SAMPLES / "borrows.json")
    events = load_events(SAMPLES / "trace.ndjson", snapshot)
    results: dict[str, Any] = {}

    if "borrow-cycle" in selected:
        copied = injection_dir / "borrows-cycle.json"
        shutil.copy2(SAMPLES / "anomalies" / "borrow_cycle" / "borrows.json", copied)
        results["borrow-cycle"] = _validate_or_capture(
            lambda: load_parameter_snapshot(SAMPLES / "pools.json", copied)
        )
        for record in results["borrow-cycle"].get("records", []):
            if record.get("business_code") == "E_BORROW":
                logger.emit("E_BORROW", record["message"], path=record["path"], line=record["line"], details=record["details"])

    if "missing-close" in selected:
        copied = injection_dir / "trace-missing-close.ndjson"
        shutil.copy2(SAMPLES / "anomalies" / "missing_close" / "trace.ndjson", copied)
        results["missing-close"] = _validate_or_capture(lambda: load_events(copied, snapshot))
        for record in results["missing-close"].get("records", []):
            if record.get("business_code") == "E_GAP":
                logger.emit("E_GAP", record["message"], path=record["path"], line=record["line"], details=record["details"])

    if "version-mismatch" in selected:
        copied_ledger = injection_dir / "version-mismatch.sqlite3"
        with sqlite3.connect(str(demo_ledger)) as source, sqlite3.connect(str(copied_ledger)) as target:
            source.backup(target)
        anomaly_snapshot = load_parameter_snapshot(
            SAMPLES / "anomalies" / "version_mismatch" / "pools.json",
            SAMPLES / "anomalies" / "version_mismatch" / "borrows.json",
        )
        try:
            with SQLiteLedger(copied_ledger) as store:
                store.initialize(anomaly_snapshot)
        except VersionMismatchError as exc:
            record = exc.records[0].to_dict()
            results["version-mismatch"] = {
                "rejected": True,
                "envelope_code": "E_VERSION",
                "records": [record],
                "business_codes": ["E_VERSION"],
            }
            logger.emit("E_VERSION", record["message"], path=record["path"], details=record["details"])
        else:
            results["version-mismatch"] = {"rejected": False}

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic gateway quota-waterline demo")
    parser.add_argument(
        "--inject",
        action="append",
        choices=["borrow-cycle", "missing-close", "version-mismatch", "all"],
        default=[],
        help="run one or more anomaly injections (default: all)",
    )
    args = parser.parse_args()
    selected = args.inject or ["all"]
    if "all" in selected:
        selected = ["borrow-cycle", "missing-close", "version-mismatch"]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for path in ARTIFACTS.iterdir():
        if path.name == "workspace.json":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    ledger_path = ARTIFACTS / "demo_ledger.sqlite3"
    crash_log_path = ARTIFACTS / "crash_run.ndjson"
    demo_log_path = ARTIFACTS / "demo_log.ndjson"
    summary_path = ARTIFACTS / "demo_summary.json"

    _print_header("ENVIRONMENT")
    print(f"python={sys.version.split()[0]}")
    print(f"numpy={numpy.__version__}")
    print("stdlib=socket,sqlite3,json")
    loopback_port = check_loopback_port(0)
    print(f"loopback=127.0.0.1:{loopback_port} available")

    _print_header("LOAD AND VALIDATE")
    initial_snapshot = load_parameter_snapshot(SAMPLES / "pools.json", SAMPLES / "borrows.json")
    events = load_events(SAMPLES / "trace.ndjson", initial_snapshot)
    reconfigs = load_reconfigurations([SAMPLES / "reconfig_v2.json"])
    print(
        f"version={initial_snapshot.version} pools={len(initial_snapshot.pools)} "
        f"borrows={len(initial_snapshot.borrow_rules)} events={len(events)} "
        f"reconfigs={len(reconfigs)}"
    )

    _print_header("FORCED TERMINATION")
    child = _run_crashing_child(ledger_path, crash_log_path)
    print(f"child_exit_code={child.returncode} (expected 137)")
    if child.returncode not in (137, -9):
        print(child.stdout)
        print(child.stderr, file=sys.stderr)
        return 1
    print("exit_report_produced=false")

    _print_header("RECOVERY, CONTINUATION, REPLAY")
    logger = StructuredLogger(demo_log_path)
    broker = LoopbackSignalBroker(0).start()
    try:
        outcome = ExperimentRunner(
            RunnerConfig(
                ledger_path=ledger_path,
                initial_snapshot=initial_snapshot,
                events=events,
                reconfigurations=reconfigs,
                batch_size=5,
            ),
            signal_broker=broker,
            logger=logger,
        ).run()
    finally:
        broker.stop()
    print(
        f"recovered_through_seq={outcome['recovery']['recovered_through_seq']} "
        f"pending_start_seq={outcome['recovery']['pending_batch']['start_seq']}"
    )
    print(f"unknown_connections={outcome['recovery']['unknown_connections']}")
    print(f"recovery_replay_consistent={outcome['replay']['consistent']} checked={outcome['replay']['events_checked']}")
    print(f"final_replay_consistent={outcome['final_replay']['consistent']} checked={outcome['final_replay']['events_checked']}")
    print(f"continued_events={outcome['committed_events']} last_event_seq={outcome['last_event_seq']}")

    with SQLiteLedger(ledger_path) as store:
        scenarios = _summarize_scenarios(store, outcome)
        stale_warning_count = int(
            store.connection.execute("SELECT COUNT(*) AS n FROM warnings WHERE batch_id=(SELECT MAX(batch_id) FROM batches) AND stale=1").fetchone()["n"]
        )
        ledger_counts = store.counts()
        ledger_meta = store.ledger_meta()

    _print_header("SCENARIO SUMMARY")
    print(
        "single_pool_full="
        f"seq3(alpha capacity {scenarios['single_pool_full']['alpha_capacity_in_use']}/3, "
        f"remaining {scenarios['single_pool_full']['alpha_remaining_connections']}), "
        f"seq7 issues={scenarios['single_pool_full']['rejected_issue_codes']}"
    )
    borrow = scenarios["cross_pool_borrow_and_return"]
    print(
        "borrow_return="
        f"{borrow['loan_id']} {borrow['origin_pool']}->{borrow['current_owner_pool']} "
        f"overdue={borrow['overdue_event_seq']} status={borrow['loan_status']}"
    )
    diagnostic = scenarios["diagnostic_at_seq_15"]
    print(
        "waterline_diagnostic="
        f"remaining={diagnostic['remaining_connections']} eta_ms={diagnostic['eta_ms']} "
        f"window_ms={diagnostic['estimation_window_ms']} error_bound_ms={diagnostic['error_bound_ms']}"
    )
    tightening = scenarios["parameter_tightening"]
    print(
        "parameter_tightening="
        f"v{tightening['old_version']}->v{tightening['new_version']} "
        f"cleared={[item['pool_id'] for item in tightening['cleared_alert_records']]} "
        f"stale_warnings={stale_warning_count} current_warnings={len(tightening['post_transition_current_warnings'])}"
    )
    print(f"final_pools={json.dumps(scenarios['final_pools'], sort_keys=True, ensure_ascii=False)}")

    _print_header("ANOMALY INJECTION")
    injections = _run_injections(selected, logger, ledger_path)
    for name in sorted(injections):
        item = injections[name]
        codes = item.get("business_codes", [])
        print(f"{name}: rejected={item.get('rejected')} business_codes={codes}")

    summary = {
        "code": "OK",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "loopback": f"127.0.0.1:{loopback_port}",
        },
        "input": {
            "parameter_version": initial_snapshot.version,
            "pools": len(initial_snapshot.pools),
            "borrow_rules": len(initial_snapshot.borrow_rules),
            "events": len(events),
        },
        "forced_termination": {
            "child_exit_code": child.returncode,
            "exit_report_produced": False,
        },
        "recovery": outcome["recovery"],
        "replay": outcome["replay"],
        "final_replay": outcome["final_replay"],
        "continuation": {
            "committed_batches": outcome["committed_batches"],
            "committed_events": outcome["committed_events"],
            "last_event_seq": outcome["last_event_seq"],
        },
        "scenarios": scenarios,
        "stale_warning_count": stale_warning_count,
        "ledger": {
            "path": str(ledger_path.relative_to(ROOT)),
            "counts": ledger_counts,
            "meta": ledger_meta,
        },
        "injections": injections,
        "logs": {
            "crash": str(crash_log_path.relative_to(ROOT)),
            "demo": str(demo_log_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_header("ARTIFACTS")
    print(f"summary={summary_path.relative_to(ROOT)}")
    print(f"ledger={ledger_path.relative_to(ROOT)}")
    print(f"log={demo_log_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

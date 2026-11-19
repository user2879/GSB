"""End-to-end deterministic demonstration and verification report."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srebudget.clock import LogicalClock
from srebudget.engine import DiagnosticEngine
from srebudget.loader import load_chain, load_trace
from srebudget.replay import ReplayVerifier
from srebudget.runner import BatchRunner
from srebudget.samples import write_samples
from srebudget.storage import SessionStore

DATA = ROOT / "data"


def _remove_sqlite(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _uninterrupted(chain: Any, trace: Any) -> dict[str, Any]:
    clock = LogicalClock(0)
    engine = DiagnosticEngine(chain, clock)
    decisions = []
    for call in trace.calls:
        clock.follow(call.timestamp_ms)
        decisions.append(engine.process_call(call))
    reasons = Counter(
        item.trigger_reason for item in decisions if item.trigger_reason is not None
    )
    return {
        "calls": len(decisions),
        "fast_failed_calls": sum(1 for item in decisions if item.fast_failed),
        "trigger_reasons": dict(sorted(reasons.items())),
        "fast_fail_chains": [
            {
                "call_index": item.call_index,
                "trigger_source": item.trigger_source,
                "trigger_reason": item.trigger_reason,
                "chain": list(item.fast_fail_chain),
            }
            for item in decisions
            if item.fast_fail_chain
        ],
        "events": list(engine.events),
        "services": engine.service_rows(clock.now_ms()),
    }


def _snapshot_replay(chain: Any, trace: Any) -> dict[str, Any]:
    db_path = DATA / "demo_replay.sqlite3"
    _remove_sqlite(db_path)
    clock = LogicalClock(0)
    engine = DiagnosticEngine(chain, clock)
    for call in trace.calls[:5]:
        clock.follow(call.timestamp_ms)
        engine.process_call(call)
    with SessionStore(db_path) as store:
        session_id = store.create_session(chain, trace, session_id="demo-replay")
        snapshot = store.create_snapshot(session_id, engine, chain, trace)
        loaded = store.load_snapshot(snapshot.snapshot_id)
        replay = ReplayVerifier(chain, trace, loaded).run()
        return {
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_cursor": snapshot.cursor,
            "content_digest": snapshot.content_digest,
            "replay": replay.as_dict(),
        }


def _restore_demo(chain: Any, trace: Any) -> dict[str, Any]:
    db_path = DATA / "demo_restore.sqlite3"
    _remove_sqlite(db_path)
    with SessionStore(db_path) as store:
        session_id = store.create_session(chain, trace, session_id="demo-restore")
        runner = BatchRunner(
            store,
            session_id,
            chain,
            trace,
            clock=LogicalClock(0),
            follow_trace_time=True,
        )
        runner.run(max_batches=1, batch_size=4)
        snapshot = runner.create_snapshot()
        runner.run(batch_size=4)
        result = store.restore_snapshot(
            snapshot.snapshot_id, runner.engine, chain, trace
        )
        return {
            "snapshot_cursor": snapshot.cursor,
            "restore": result.as_dict(),
            "valid_conclusions_after_restore": len(
                store.list_conclusions(session_id, valid_only=True)
            ),
        }


def _crash_recovery(chain: Any, trace: Any) -> dict[str, Any]:
    db_path = DATA / "demo_crash.sqlite3"
    _remove_sqlite(db_path)
    with SessionStore(db_path) as store:
        session_id = store.create_session(chain, trace, session_id="demo-crash")
        runner = BatchRunner(
            store,
            session_id,
            chain,
            trace,
            clock=LogicalClock(0),
            follow_trace_time=True,
        )
        runner.run(max_batches=1, batch_size=4)
        command = [
            sys.executable,
            str(ROOT / "examples" / "crash_worker.py"),
            "--db",
            str(db_path),
            "--chain",
            str(DATA / "chain.json"),
            "--trace",
            str(DATA / "trace.ndjson"),
            "--session-id",
            session_id,
            "--start",
            "5",
            "--end",
            "9",
        ]
        killed = subprocess.run(command, cwd=ROOT, check=False)

    with SessionStore(db_path) as reopened:
        runner = BatchRunner(
            reopened,
            session_id,
            chain,
            trace,
            clock=LogicalClock(0),
            follow_trace_time=True,
        )
        recovery = runner.recovery
        outcome = runner.run(batch_size=4)
        return {
            "worker_exit_code": killed.returncode,
            "recovery": recovery.as_dict(),
            "rerun_processed_calls": outcome.processed_calls,
            "final_cursor": runner.engine.cursor,
            "unknown_history": reopened.list_unknown(session_id),
        }


def _injections() -> list[dict[str, Any]]:
    reports = []
    for kind in ("chain_corrupt", "timestamp_regression", "snapshot_digest_mismatch"):
        command = [
            sys.executable,
            "-m",
            "srebudget.cli",
            "inject",
            kind,
            "--data-dir",
            str(DATA),
        ]
        completed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        payload = json.loads(completed.stdout)
        reports.append(
            {
                "kind": kind,
                "exit_code": completed.returncode,
                "reported_code": payload["reported"]["code"],
                "details": payload["reported"]["details"],
            }
        )
    return reports


def main() -> int:
    chain_path, trace_path = write_samples(DATA)
    chain = load_chain(chain_path)
    trace = load_trace(trace_path, chain)
    report = {
        "ok": True,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "inputs": {
            "chain_version": chain.version,
            "chain_digest": chain.digest,
            "trace_digest": trace.content_digest,
            "calls": len(trace),
        },
        "uninterrupted": _uninterrupted(chain, trace),
        "snapshot_replay": _snapshot_replay(chain, trace),
        "whole_state_restore": _restore_demo(chain, trace),
        "forced_termination_recovery": _crash_recovery(chain, trace),
        "injections": _injections(),
    }
    report_path = DATA / "demo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "report": str(report_path),
                "snapshot_replay": report["snapshot_replay"]["replay"],
                "unknown_batches": len(
                    report["forced_termination_recovery"]["recovery"][
                        "unknown_batches"
                    ]
                ),
                "injections": [
                    (item["kind"], item["reported_code"])
                    for item in report["injections"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

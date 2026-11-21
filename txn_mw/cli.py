"""Command-line entry points for server, ledger tools, verification and Tk."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_json
from .engine import Engine, EngineConfig
from .errors import DuplicateTxn, MWError
from .ledger import inspect_ledger, load_ledger, replace_ledger
from .logging import StructuredLogger
from .server import RunningServer, request
from .store import Store
from .verify import verify_database, verify_ledger_file


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "txn_mw.db"
DEFAULT_LOG = ROOT / "artifacts" / "structured.jsonl"


def _config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        query_budget_ms=args.query_budget_ms,
        pull_budget_ms=args.pull_budget_ms,
        query_default_max=args.query_max,
        query_retry_ms=args.query_retry_ms,
        query_tick_ms=args.query_tick_ms,
        query_workers=args.query_workers,
        idem_max_entries=args.idem_max_entries,
        idem_ttl_ms=args.idem_ttl_ms,
    )


def inject_tail_truncation(db_path: str | Path) -> bool:
    path = Path(db_path)
    if not path.exists():
        return False
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT row_id FROM events ORDER BY event_seq DESC LIMIT 1").fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE events SET record=substr(record,1,length(record)-1) WHERE row_id=?",
            (row[0],),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def _apply_post_recovery_faults(engine: Engine, faults: list[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if "duplicate_txn" in faults:
        txn_id = "__fault_duplicate__"
        half = {
            "txn_id": txn_id,
            "version": 1,
            "body": {"injected_fault": "duplicate_txn"},
            "txn_timeout_ms": 60_000,
            "query": {"host": "127.0.0.1", "port": 9},
            "query_max": 3,
        }
        try:
            engine.persist_half(half)
        except DuplicateTxn:
            pass
        try:
            engine.persist_half(half)
            reports.append({"fault": "duplicate_txn", "occurred": False, "reason": "unexpectedly accepted"})
        except DuplicateTxn as exc:
            reports.append({"fault": "duplicate_txn", "occurred": True, "code": "E_DUP", "txn_id": txn_id})
    if "query_command_conflict" in faults:
        reports.append({"fault": "query_command_conflict", **engine.inject_conflict_fault()})
    return reports


def command_serve(args: argparse.Namespace) -> int:
    db = Path(args.db).resolve()
    logger = StructuredLogger(args.log)
    faults = [item for item in (args.inject_fault or []) if item != "none"]
    if "log_tail_truncate" in faults:
        if not inject_tail_truncation(db):
            logger.error(
                "E_LOG",
                "log_tail_truncate injection had no event record to truncate",
                txn_id=None,
                operation="startup_injection",
            )
    engine = Engine(db, config=_config(args), logger=logger)
    fault_reports = _apply_post_recovery_faults(engine, faults)
    running = RunningServer(engine, args.host, args.port)
    running.start()
    host, port = running.address
    port_info = {
        "ok": True,
        "service": "txn-mw",
        "host": host,
        "port": port,
        "db": str(db),
        "format_version": FORMAT_VERSION,
        "recovery_report": engine.recovery_report,
        "fault_reports": fault_reports,
    }
    if args.port_file:
        path = Path(args.port_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(canonical_json(port_info) + "\n", encoding="utf-8")
        os.replace(temp, path)
    print(canonical_json(port_info), flush=True)

    stop = threading.Event()

    def handle_signal(signum: int, frame: Any) -> None:
        stop.set()

    previous: dict[int, Any] = {}
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                previous[sig] = signal.signal(sig, handle_signal)
            except (ValueError, OSError):
                pass
    try:
        while not stop.is_set():
            stop.wait(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        report = running.stop(report=True)
        if args.exit_report and report is not None:
            path = Path(args.exit_report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical_json(report) + "\n", encoding="utf-8")
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
    return 0


def command_export(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        result = store.export_ledger(args.out)
    finally:
        store.close()
    print(canonical_json(result))
    return 0


def command_restore(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    store = Store(args.db)
    try:
        result = replace_ledger(store, args.ledger, logger=logger)
    finally:
        store.close()
    print(canonical_json(result))
    return 0 if result.get("ok") else 1


def command_verify(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    if args.ledger:
        result = verify_ledger_file(args.ledger, logger=logger)
    else:
        store = Store(args.db)
        try:
            result = verify_database(store, logger=logger)
        finally:
            store.close()
    print(canonical_json(result))
    return 0 if result.get("ok") else 1


def command_recover_report(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    engine = Engine(args.db, config=_config(args), logger=logger, start_worker=False)
    report = engine.recovery_report
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    engine.store.close()
    return 0


def command_inspect_ledger(args: argparse.Namespace) -> int:
    value = load_ledger(args.ledger)
    print(canonical_json({"ok": True, "inspection": inspect_ledger(value)}))
    return 0


def command_send(args: argparse.Namespace) -> int:
    payload = json.loads(args.json)
    print(canonical_json(request(args.host, args.port, payload, timeout=args.timeout)))
    return 0


def command_scenario(args: argparse.Namespace) -> int:
    from .scenario import run_scenario

    result = run_scenario(args.scenario, artifacts_dir=args.artifacts, log_path=args.log)
    print(canonical_json(result))
    return 0 if result.get("ok") else 1


def command_tui(args: argparse.Namespace) -> int:
    try:
        from .tui import launch
        return launch(args.db, args.log)
    except ImportError:
        print(
            "tkinter is unavailable in this Python installation. Install the official Python "
            "distribution with the tcl/tk option, or use the CLI/server mode.",
            file=sys.stderr,
        )
        return 2


def add_engine_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-budget-ms", type=int, default=250)
    parser.add_argument("--pull-budget-ms", type=int, default=200)
    parser.add_argument("--query-max", type=int, default=3)
    parser.add_argument("--query-retry-ms", type=int, default=100)
    parser.add_argument("--query-tick-ms", type=int, default=20)
    parser.add_argument("--query-workers", type=int, default=8)
    parser.add_argument("--idem-max-entries", type=int, default=1024)
    parser.add_argument("--idem-ttl-ms", type=int, default=60_000)
    parser.add_argument("--log", default=str(DEFAULT_LOG))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="txn-mw",
        description="Loopback transactional-message middleware with durable query arbitration and idempotent pulls.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the TCP JSON-lines server")
    serve.add_argument("--db", default=str(DEFAULT_DB))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--port-file")
    serve.add_argument("--exit-report")
    serve.add_argument(
        "--inject-fault",
        action="append",
        choices=["none", "log_tail_truncate", "duplicate_txn", "query_command_conflict"],
        default=[],
    )
    add_engine_options(serve)
    serve.set_defaults(func=command_serve)

    export = sub.add_parser("export-ledger", help="write a digest-protected JSON ledger")
    export.add_argument("--db", default=str(DEFAULT_DB))
    export.add_argument("--out", required=True)
    export.set_defaults(func=command_export)

    restore = sub.add_parser("restore-ledger", help="whole-replace state from a ledger with a read-only backup")
    restore.add_argument("--db", default=str(DEFAULT_DB))
    restore.add_argument("--ledger", required=True)
    restore.add_argument("--log", default=str(DEFAULT_LOG))
    restore.set_defaults(func=command_restore)

    verify = sub.add_parser("verify", help="independently replay each transaction")
    verify.add_argument("--db", default=str(DEFAULT_DB))
    verify.add_argument("--ledger")
    verify.add_argument("--log", default=str(DEFAULT_LOG))
    verify.set_defaults(func=command_verify)

    inspect = sub.add_parser("inspect-ledger", help="show ledger version and digest status")
    inspect.add_argument("--ledger", required=True)
    inspect.set_defaults(func=command_inspect_ledger)

    recover = sub.add_parser("recover-report", help="open SQLite and emit a recovery report")
    recover.add_argument("--db", default=str(DEFAULT_DB))
    recover.add_argument("--out")
    add_engine_options(recover)
    recover.set_defaults(func=command_recover_report)

    send = sub.add_parser("send", help="send one JSON request to a running server")
    send.add_argument("--host", default="127.0.0.1")
    send.add_argument("--port", type=int, required=True)
    send.add_argument("--json", required=True)
    send.add_argument("--timeout", type=float, default=3.0)
    send.set_defaults(func=command_send)

    scenario = sub.add_parser("scenario", help="run deterministic loopback acceptance samples")
    scenario.add_argument("--scenario", default=str(ROOT / "samples" / "scenario.json"))
    scenario.add_argument("--artifacts", default=str(ROOT / "artifacts" / "scenario"))
    scenario.add_argument("--log", default=str(ROOT / "artifacts" / "scenario-structured.jsonl"))
    scenario.set_defaults(func=command_scenario)

    tui = sub.add_parser("tui", help="optional Tk control panel (requires tcl/tk)")
    tui.add_argument("--db", default=str(DEFAULT_DB))
    tui.add_argument("--log", default=str(DEFAULT_LOG))
    tui.set_defaults(func=command_tui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MWError as exc:
        StructuredLogger(getattr(args, "log", None)).error(
            exc.code, exc.message, path=exc.path, details=exc.details
        )
        print(canonical_json(exc.as_dict()))
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

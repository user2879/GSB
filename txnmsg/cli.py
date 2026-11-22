"""Command-line interface for demo, diagnostics, snapshots and injections."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_json
from .demo import CRASH_EXIT_CODE, run_demo
from .diagnostics import build_diagnostics, render_text
from .errors import BusinessError, SpecError
from .injections import (
    inject_corrupt_batch,
    inject_duplicate_position,
    inject_ledger_digest,
    inject_snapshot_digest,
)
from .observability import StructuredLogger
from .replay_verify import replay_verify
from .snapshots import (
    create_snapshot,
    list_snapshots,
    load_snapshot,
    next_snapshot_path,
    restore_snapshot,
)
from .store import Ledger
from .ui import TkUnavailable, run_ui, tkinter_available
from .validation import load_partitions, load_transactions


def _database(workdir: Path) -> Path:
    return workdir / "ledger.sqlite"


def _logger(workdir: Path) -> StructuredLogger:
    return StructuredLogger(workdir / "logs" / "business.jsonl")


def _print_json(payload: Any) -> None:
    print(canonical_json(payload))


def _add_workdir(parser: argparse.ArgumentParser, default: str = "demo_workspace") -> None:
    parser.add_argument("--workdir", default=default, help="working directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="txnmsg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="run deterministic end-to-end demo")
    _add_workdir(demo_parser)
    demo_parser.add_argument(
        "--quiet", action="store_true", help="print only the summary path and status"
    )

    view_parser = subparsers.add_parser("view", help="print stable read-only diagnostics")
    _add_workdir(view_parser)
    view_parser.add_argument("--database")
    view_parser.add_argument("--json", action="store_true", help="emit JSON rather than text")

    ui_parser = subparsers.add_parser("ui", help="launch Tkinter diagnostics UI")
    _add_workdir(ui_parser)
    ui_parser.add_argument("--database")

    validate_parser = subparsers.add_parser("validate", help="validate local JSON inputs")
    validate_parser.add_argument("--partitions", required=True)
    validate_parser.add_argument("--transactions", required=True)
    _add_workdir(validate_parser)

    snapshot_parser = subparsers.add_parser("snapshot", help="create a snapshot")
    _add_workdir(snapshot_parser)
    snapshot_parser.add_argument("--database")
    snapshot_parser.add_argument("--output")

    restore_parser = subparsers.add_parser("restore", help="restore a snapshot wholesale")
    _add_workdir(restore_parser)
    restore_parser.add_argument("--database")
    restore_parser.add_argument("--snapshot", required=True)

    replay_parser = subparsers.add_parser("replay", help="run post-restore replay verification")
    _add_workdir(replay_parser)
    replay_parser.add_argument("--database")
    replay_parser.add_argument("--snapshot")

    inject_parser = subparsers.add_parser("inject", help="inject an input-only anomaly")
    _add_workdir(inject_parser)
    inject_parser.add_argument(
        "--kind",
        required=True,
        choices=("corrupt-batch", "duplicate-position", "ledger-digest", "snapshot-digest"),
    )
    inject_parser.add_argument("--database")
    inject_parser.add_argument("--snapshot")

    crash_parser = subparsers.add_parser("_crash-worker")
    crash_parser.add_argument("--database", required=True)
    crash_parser.add_argument("--transaction-id", required=True)
    crash_parser.add_argument("--state", required=True)
    return parser


def _command_demo(args: argparse.Namespace) -> int:
    result = run_demo(args.workdir)
    if args.quiet:
        print(f"DEMO {result['status']} summary={result['summary_path']}")
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _command_view(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    database = Path(args.database) if args.database else _database(workdir)
    with Ledger(database, auto_recover=False, readonly=True, logger=_logger(workdir)) as ledger:
        view = build_diagnostics(ledger, workdir)
    if args.json:
        _print_json({"status": "OK", "view": view})
    else:
        sys.stdout.write(render_text(view))
    return 0


def _command_ui(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    database = Path(args.database) if args.database else _database(workdir)
    try:
        run_ui(database, workdir)
        return 0
    except TkUnavailable as exc:
        with Ledger(database, auto_recover=False, readonly=True, logger=_logger(workdir)) as ledger:
            view = build_diagnostics(ledger, workdir)
        print(f"UI_UNAVAILABLE {exc}", file=sys.stderr)
        print(canonical_json({"status": "FALLBACK", "tkinter_available": tkinter_available(), "view": view}))
        return 0


def _command_validate(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    logger = _logger(workdir)
    partitions = load_partitions(args.partitions, logger)
    transactions = load_transactions(args.transactions, partitions, logger)
    _print_json(
        {
            "status": "OK",
            "partition_count": len(partitions),
            "transaction_count": len(transactions),
            "partition_ids": [partition.partition_id for partition in partitions],
            "transaction_ids": [transaction.transaction_id for transaction in transactions],
        }
    )
    return 0


def _command_snapshot(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    database = Path(args.database) if args.database else _database(workdir)
    output = Path(args.output) if args.output else next_snapshot_path(workdir)
    with Ledger(database, logger=_logger(workdir)) as ledger:
        snapshot = create_snapshot(ledger, output)
    _print_json(
        {
            "status": "OK",
            "snapshot": snapshot.path,
            "ledger_digest": snapshot.ledger_digest,
            "content_digest": snapshot.content_digest,
            "event_cursor": snapshot.event_cursor,
        }
    )
    return 0


def _command_restore(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    database = Path(args.database) if args.database else _database(workdir)
    with Ledger(database, logger=_logger(workdir)) as ledger:
        result = restore_snapshot(ledger, args.snapshot, logger=_logger(workdir))
    _print_json({"status": "OK", **result.as_dict()})
    return 0


def _command_replay(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    database = Path(args.database) if args.database else _database(workdir)
    with Ledger(database, auto_recover=False, readonly=True, logger=_logger(workdir)) as ledger:
        report = replay_verify(ledger, args.snapshot, logger=_logger(workdir))
    _print_json({"status": "OK", **report.as_dict()})
    return 0


def _command_inject(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    logger = _logger(workdir)
    database = Path(args.database) if args.database else _database(workdir)
    try:
        if args.kind == "corrupt-batch":
            path = inject_corrupt_batch(workdir)
            partitions = load_partitions(workdir / "samples" / "partitions.json", logger)
            load_transactions(path, partitions, logger)
        elif args.kind == "duplicate-position":
            path = inject_duplicate_position(workdir)
            partitions = load_partitions(workdir / "samples" / "partitions.json", logger)
            load_transactions(path, partitions, logger)
        elif args.kind == "ledger-digest":
            path = inject_ledger_digest(database, workdir)
            Ledger(path, auto_recover=False, logger=logger)
        else:
            snapshots = list_snapshots(workdir)
            source = Path(args.snapshot) if args.snapshot else (
                Path(snapshots[-1].path) if snapshots else None
            )
            if source is None:
                raise SpecError("没有可用于注入的快照", field="snapshot")
            path = inject_snapshot_digest(source, workdir)
            load_snapshot(path)
        raise RuntimeError("injected anomaly was unexpectedly accepted")
    except BusinessError as error:
        logger.write_error(error)
        _print_json(
            {
                "status": "INJECTED",
                "input": path if "path" in locals() else None,
                **error.to_dict(),
            }
        )
        return error.exit_code


def _command_crash_worker(args: argparse.Namespace) -> int:
    database = Path(args.database)
    with Ledger(database) as ledger:
        ledger.advance_state(
            args.transaction_id,
            args.state,
            crash_hook=lambda: os._exit(CRASH_EXIT_CODE),
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            return _command_demo(args)
        if args.command == "view":
            return _command_view(args)
        if args.command == "ui":
            return _command_ui(args)
        if args.command == "validate":
            return _command_validate(args)
        if args.command == "snapshot":
            return _command_snapshot(args)
        if args.command == "restore":
            return _command_restore(args)
        if args.command == "replay":
            return _command_replay(args)
        if args.command == "inject":
            return _command_inject(args)
        if args.command == "_crash-worker":
            return _command_crash_worker(args)
    except BusinessError as error:
        workdir = Path(getattr(args, "workdir", "demo_workspace")).resolve()
        logger = _logger(workdir)
        logger.write_error(error)
        _print_json({"status": "ERROR", **error.to_dict()})
        return error.exit_code
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "ERROR",
                    "code": "E_INTERNAL",
                    "message": str(error),
                    "exception": type(error).__name__,
                }
            )
        )
        return 1
    parser.error("unknown command")
    return 1

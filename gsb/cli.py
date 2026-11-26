"""Command line interface for validation, execution, recovery and audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .invalidation import audit_invalidation, compare_ledger_with_rebuild, full_rebuild
from .ledger import Ledger
from .spec import load_session


def _common_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="combined session JSON file")
    parser.add_argument("--shards", help="shards JSON document (separate-file mode)")
    parser.add_argument("--checkpoints", help="checkpoints JSON document")
    parser.add_argument("--batches", help="batches JSON document")
    parser.add_argument("--events", help="events JSON document")
    parser.add_argument("--global-budget", type=int, help="global resource units")


def _load_from_args(args: argparse.Namespace):
    return load_session(
        config_path=args.config,
        shards_path=args.shards,
        checkpoints_path=args.checkpoints,
        batches_path=args.batches,
        events_path=args.events,
        global_budget=args.global_budget,
    )


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _validate_command(args: argparse.Namespace) -> dict[str, Any]:
    session = _load_from_args(args)
    return {
        "ok": True,
        "code": "OK",
        "source": session.source,
        "format_version": session.format_version,
        "shard_count": len(session.shards),
        "checkpoint_count": len(session.checkpoints),
        "batch_count": len(session.batches),
        "event_count": len(session.events),
        "global_budget": session.global_budget,
        "shard_set_version": session.shard_set_version,
    }


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    from .engine import run_session

    session = _load_from_args(args)
    return run_session(
        session,
        ledger_path=args.ledger,
        startup_timeout_s=args.startup_timeout_ms / 1000.0,
        handshake_timeout_s=args.handshake_timeout_ms / 1000.0,
        connect_timeout_s=args.connect_timeout_ms / 1000.0,
        read_timeout_s=args.read_timeout_ms / 1000.0,
        max_workers=args.max_workers,
    )


def _recover_command(args: argparse.Namespace) -> dict[str, Any]:
    session = _load_from_args(args)
    ledger = Ledger.open(args.ledger, session)
    try:
        recovery = ledger.recover(now_ms=0)
        replay = ledger.replay()
        return {
            "ok": True,
            "ledger": args.ledger,
            "recovery": recovery,
            "replay": replay,
            "snapshot": ledger.snapshot_info(),
        }
    finally:
        ledger.close()


def _rebuild_command(args: argparse.Namespace) -> dict[str, Any]:
    session = _load_from_args(args)
    ledger = Ledger.open(args.ledger, session)
    try:
        consistency = compare_ledger_with_rebuild(session, ledger)
        rebuilt = full_rebuild(session)
        return {
            "ok": True,
            "consistency": consistency,
            "rebuild_digest": rebuilt["rebuild_digest"],
            "batch_count": rebuilt["batch_count"],
        }
    finally:
        ledger.close()


def _invalidate_command(args: argparse.Namespace) -> dict[str, Any]:
    old = load_session(config_path=args.old)
    new = load_session(config_path=args.new)
    report = audit_invalidation(old, new)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {"ok": True, **report}


def _full_rebuild_command(args: argparse.Namespace) -> dict[str, Any]:
    session = _load_from_args(args)
    rebuilt = full_rebuild(session)
    if args.output:
        Path(args.output).write_text(
            json.dumps(rebuilt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {"ok": True, **rebuilt}


def _gui_command(args: argparse.Namespace) -> dict[str, Any]:
    from .gui import show_dashboard

    session = _load_from_args(args)
    show_dashboard(session, ledger_path=args.ledger)
    return {"ok": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsb",
        description="Local deterministic training-shard checkpoint diagnostics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="strictly validate declarations without starting workers")
    _common_spec_arguments(validate)
    validate.set_defaults(handler=_validate_command)

    run = sub.add_parser("run", help="start shard workers, schedule batches and commit the ledger")
    _common_spec_arguments(run)
    run.add_argument("--ledger", required=True, help="SQLite ledger path")
    run.add_argument("--max-workers", type=int, default=None)
    run.add_argument("--startup-timeout-ms", type=int, default=8000)
    run.add_argument("--handshake-timeout-ms", type=int, default=3000)
    run.add_argument("--connect-timeout-ms", type=int, default=2000)
    run.add_argument("--read-timeout-ms", type=int, default=30000)
    run.set_defaults(handler=_run_command)

    recover = sub.add_parser("recover", help="recover uncommitted attempts and replay committed batches")
    _common_spec_arguments(recover)
    recover.add_argument("--ledger", required=True)
    recover.set_defaults(handler=_recover_command)

    rebuild = sub.add_parser("rebuild-check", help="compare a ledger with a from-scratch rebuild")
    _common_spec_arguments(rebuild)
    rebuild.add_argument("--ledger", required=True)
    rebuild.set_defaults(handler=_rebuild_command)

    invalidate = sub.add_parser("invalidate", help="diff two configs and propagate invalid artifacts")
    invalidate.add_argument("--old", required=True)
    invalidate.add_argument("--new", required=True)
    invalidate.add_argument("--output")
    invalidate.set_defaults(handler=_invalidate_command)

    rebuild_all = sub.add_parser("rebuild", help="emit a from-scratch deterministic rebuild")
    _common_spec_arguments(rebuild_all)
    rebuild_all.add_argument("--output")
    rebuild_all.set_defaults(handler=_full_rebuild_command)

    gui = sub.add_parser("gui", help="open the optional Tkinter dashboard")
    _common_spec_arguments(gui)
    gui.add_argument("--ledger")
    gui.set_defaults(handler=_gui_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except DiagnosticError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return {
            "E_SPEC": 2,
            "E_PORT": 3,
            "E_TIMEOUT": 4,
            "E_FIELD": 5,
            "E_LEASE": 6,
            "E_REPLAY": 7,
            "E_QUOTA": 8,
            "E_UI": 9,
        }.get(exc.code, 1)
    except KeyboardInterrupt:
        print(json.dumps({"code": "E_INTERRUPTED", "message": "interrupted"}, sort_keys=True), file=sys.stderr)
        return 130
    _print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .engine import ArbiterEngine
from .errors import ArbiterError
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .snapshot import replay_ledger, restore_workspace, write_snapshot
from .spec import load_specs

DEFAULT_ENV = "samples/environments.json"
DEFAULT_VIEWS = "samples/views.json"
DEFAULT_DB = "runtime/arbiter-ledger.sqlite"
DEFAULT_SNAPSHOT = "runtime/arbiter-snapshot.json"
DEFAULT_LOG = "logs/arbiter.jsonl"
INJECTED_BAD_DIGEST = "0" * 64


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env", default=DEFAULT_ENV, help="environment declaration JSON")
    common.add_argument("--views", default=DEFAULT_VIEWS, help="view declaration JSON")
    common.add_argument("--db", default=DEFAULT_DB, help="SQLite ledger path")
    common.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="session snapshot JSON")
    common.add_argument("--log", default=DEFAULT_LOG, help="structured JSONL log")
    common.add_argument(
        "--inject-spec-corrupt",
        action="store_true",
        help="inject an invalid environment declaration and expect E_SPEC",
    )
    common.add_argument(
        "--inject-priority-ref-missing-env",
        action="store_true",
        help="inject a missing environment reference in priority bias and expect E_REF",
    )
    common.add_argument(
        "--inject-ledger-digest-mismatch",
        action="store_true",
        help="compare the ledger with an intentionally wrong digest and expect E_SNAPSHOT",
    )

    parser = argparse.ArgumentParser(
        prog="config-arbiter",
        description="Cross-environment configuration arbitration and recovery diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", parents=[common], help="arbitrate changed views and commit the ledger")
    subparsers.add_parser("status", parents=[common], help="show read-only ledger/view status")
    subparsers.add_parser("restore", parents=[common], help="restore a snapshot and replay committed views")
    subparsers.add_parser("replay", parents=[common], help="recompute committed views and verify ledger equality")
    subparsers.add_parser("snapshot", parents=[common], help="write a snapshot from the current ledger")
    subparsers.add_parser("gui", parents=[common], help="open the Tkinter diagnostic application")
    return parser


def _load_specs(args: argparse.Namespace):
    return load_specs(
        args.env,
        args.views,
        inject_spec_corrupt=args.inject_spec_corrupt,
        inject_priority_ref_missing_env=args.inject_priority_ref_missing_env,
    )


def _expected_digest(args: argparse.Namespace) -> str | None:
    return INJECTED_BAD_DIGEST if args.inject_ledger_digest_mismatch else None


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _run(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    specs = _load_specs(args)
    ledger = Ledger(args.db)
    try:
        engine = ArbiterEngine(specs, ledger)
        report = engine.run_all(expected_digest=_expected_digest(args))
        snapshot = write_snapshot(args.snapshot, specs, ledger)
        return {
            "status": "ok",
            "run": report,
            "snapshot": {
                "path": str(Path(args.snapshot)),
                "content_digest": snapshot["content_digest"],
                "completed_views": snapshot["completed_view_ids"],
            },
        }
    finally:
        ledger.close()


def _status(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    specs = _load_specs(args)
    ledger = Ledger(args.db)
    try:
        ledger.verify_integrity(expected_digest=_expected_digest(args))
        engine = ArbiterEngine(specs, ledger)
        return {"status": "ok", "report": engine.status_report()}
    finally:
        ledger.close()


def _restore(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    report = restore_workspace(
        snapshot_path=args.snapshot,
        environment_path=args.env,
        views_path=args.views,
        db_path=args.db,
        expected_ledger_digest=_expected_digest(args),
    )
    return {"status": "ok", "restore": report}


def _replay(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    specs = _load_specs(args)
    ledger = Ledger(args.db)
    try:
        ledger.verify_integrity(expected_digest=_expected_digest(args))
        engine = ArbiterEngine(specs, ledger)
        report = replay_ledger(engine)
        return {"status": "ok", "replay": report}
    finally:
        ledger.close()


def _snapshot(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    specs = _load_specs(args)
    ledger = Ledger(args.db)
    try:
        ledger.verify_integrity(expected_digest=_expected_digest(args))
        payload = write_snapshot(args.snapshot, specs, ledger)
        return {
            "status": "ok",
            "snapshot": {
                "path": str(Path(args.snapshot)),
                "content_digest": payload["content_digest"],
                "completed_views": payload["completed_view_ids"],
            },
        }
    finally:
        ledger.close()


def _gui(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    from .ui import launch_gui

    launch_gui(
        environment_path=args.env,
        views_path=args.views,
        db_path=args.db,
        snapshot_path=args.snapshot,
        log_path=args.log,
    )
    return {"status": "ok", "gui": "closed"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(args.log)
    injections = {
        "spec_corrupt": bool(args.inject_spec_corrupt),
        "priority_ref_missing_env": bool(args.inject_priority_ref_missing_env),
        "ledger_digest_mismatch": bool(args.inject_ledger_digest_mismatch),
    }
    logger.log("command_started", command=args.command, injections=injections)
    handlers = {
        "run": _run,
        "status": _status,
        "restore": _restore,
        "replay": _replay,
        "snapshot": _snapshot,
        "gui": _gui,
    }
    try:
        result = handlers[args.command](args, logger)
    except ArbiterError as exc:
        payload = {"status": "error", **exc.as_dict()}
        logger.log(
            "command_failed",
            level="ERROR",
            command=args.command,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            injections=injections,
        )
        _print_json(payload)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.log(
            "command_failed",
            level="ERROR",
            command=args.command,
            code="E_INTERNAL",
            message=str(exc),
            details={"type": type(exc).__name__},
            injections=injections,
        )
        _print_json({"status": "error", "code": "E_INTERNAL", "message": str(exc), "details": {}})
        return 1
    logger.log("command_succeeded", command=args.command, result_status=result.get("status", "ok"))
    _print_json(result)
    return 0

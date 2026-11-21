from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .engine import (
    EngineConfig,
    recover_incomplete,
    replay_snapshot,
    review_current,
    run_adjudication,
    view_current,
)
from .errors import ArbiterError
from .jsonlog import JsonlLogger
from .ledger import LedgerDB
from .spec import load_dataset
from .ui import run_gui


def _add_concurrency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--global-concurrency", type=int, default=3)
    parser.add_argument("--per-node-concurrency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-timeout-ms", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--handshake-timeout-ms", type=int, default=5000)


def _config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        global_concurrency=args.global_concurrency,
        per_node_concurrency=args.per_node_concurrency,
        batch_size=args.batch_size,
        request_timeout_ms=args.request_timeout_ms,
        retries=args.retries,
        handshake_timeout_ms=args.handshake_timeout_ms,
        fault_after_commits=getattr(args, "fault_after_commits", None),
    )


def _default_log_path(db_path: str) -> str:
    path = Path(db_path)
    if str(path) == ":memory:":
        return "trace_arbiter.log.jsonl"
    return str(path.with_suffix(path.suffix + ".log.jsonl"))


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _require_no_incomplete_run(db: LedgerDB) -> None:
    run = db.incomplete_run()
    if run is not None:
        raise ArbiterError(
            "E_RECOVERY",
            "an incomplete run exists; use the recover command instead of starting a new run",
            details={
                "run_id": run["run_id"],
                "status": run["status"],
                "progress_cursor": run["progress_cursor"],
                "batch_count": run["batch_count"],
            },
        )


def _cmd_adjudicate(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    inject = args.inject if args.inject in ("corrupt-message", "parent-missing") else None
    dataset = load_dataset(args.nodes, args.spans, inject=inject)
    db = LedgerDB(args.db)
    try:
        if args.inject == "snapshot-mismatch":
            snapshot = db.get_snapshot()
            db.verify_snapshot_digest(snapshot["snapshot_id"], inject_mismatch=True)
        _require_no_incomplete_run(db)
        result = run_adjudication(db, dataset, _config_from_args(args), logger)
        result["ledger_file"] = str(Path(args.db).resolve())
        return result
    finally:
        db.close()


def _cmd_review(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    db = LedgerDB(args.db)
    try:
        if args.inject == "snapshot-mismatch":
            snapshot = db.get_snapshot()
            db.verify_snapshot_digest(snapshot["snapshot_id"], inject_mismatch=True)
        return review_current(db, _config_from_args(args), logger)
    finally:
        db.close()


def _cmd_replay(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    db = LedgerDB(args.db)
    try:
        return replay_snapshot(
            db,
            _config_from_args(args),
            logger,
            inject_snapshot_mismatch=args.inject == "snapshot-mismatch",
        )
    finally:
        db.close()


def _cmd_recover(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    db = LedgerDB(args.db)
    try:
        return recover_incomplete(
            db,
            _config_from_args(args),
            logger,
            replay_after=not args.no_replay,
        )
    finally:
        db.close()


def _cmd_view(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    db = LedgerDB(args.db)
    try:
        if args.inject == "snapshot-mismatch":
            snapshot = db.get_snapshot()
            db.verify_snapshot_digest(snapshot["snapshot_id"], inject_mismatch=True)
        if args.gui:
            run_gui(db)
            return {"ok": True, "operation": "view", "mode": "gui"}
        started_view = view_current(db)
        logger.emit(
            "cache_view_complete",
            snapshot_id=started_view["snapshot_id"],
            query_elapsed_us=started_view["query_elapsed_us"],
        )
        return started_view
    finally:
        db.close()


def _cmd_verify_ledger(args: argparse.Namespace, logger: JsonlLogger) -> dict[str, Any]:
    db = LedgerDB(args.db)
    try:
        result = db.verify_ledger_entries()
        logger.emit("ledger_entries_verified", **result)
        return result
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-arbiter",
        description="Local deterministic distributed-trace arbitration ledger",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    adjudicate = subparsers.add_parser("adjudicate", help="run a full arbitration round")
    adjudicate.add_argument("--nodes", required=True)
    adjudicate.add_argument("--spans", required=True)
    adjudicate.add_argument("--db", required=True)
    adjudicate.add_argument("--log")
    adjudicate.add_argument(
        "--inject",
        choices=("none", "corrupt-message", "parent-missing", "snapshot-mismatch"),
        default="none",
    )
    adjudicate.add_argument("--fault-after-commits", type=int, help=argparse.SUPPRESS)
    _add_concurrency_args(adjudicate)
    adjudicate.set_defaults(handler=_cmd_adjudicate)

    review = subparsers.add_parser("review", help="authoritatively recompute the current ledger")
    review.add_argument("--db", required=True)
    review.add_argument("--log")
    review.add_argument(
        "--inject",
        choices=("none", "snapshot-mismatch"),
        default="none",
    )
    _add_concurrency_args(review)
    review.set_defaults(handler=_cmd_review)

    replay = subparsers.add_parser("replay", help="snapshot replay and byte comparison")
    replay.add_argument("--db", required=True)
    replay.add_argument("--log")
    replay.add_argument(
        "--inject",
        choices=("none", "snapshot-mismatch"),
        default="none",
    )
    _add_concurrency_args(replay)
    replay.set_defaults(handler=_cmd_replay)

    recover = subparsers.add_parser("recover", help="resume the newest incomplete run")
    recover.add_argument("--db", required=True)
    recover.add_argument("--log")
    recover.add_argument("--no-replay", action="store_true")
    _add_concurrency_args(recover)
    recover.set_defaults(handler=_cmd_recover)

    view = subparsers.add_parser("view", help="read the current cached ledger view")
    view.add_argument("--db", required=True)
    view.add_argument("--log")
    view.add_argument("--json", action="store_true", help="headless JSON output")
    view.add_argument("--gui", action="store_true", help="open the Tkinter quick view")
    view.add_argument(
        "--inject",
        choices=("none", "snapshot-mismatch"),
        default="none",
    )
    view.set_defaults(handler=_cmd_view)

    verify = subparsers.add_parser("verify-ledger", help="verify every current entry digest")
    verify.add_argument("--db", required=True)
    verify.add_argument("--log")
    verify.set_defaults(handler=_cmd_verify_ledger)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_path = args.log or _default_log_path(args.db)
    with JsonlLogger(log_path) as logger:
        try:
            result = args.handler(args, logger)
        except ArbiterError as exc:
            logger.error("command_failed", exc)
            print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return exc.exit_code
        except KeyboardInterrupt:
            error = ArbiterError("E_RECOVERY", "interrupted; committed batches may be recovered")
            logger.error("command_interrupted", error)
            print(json.dumps(error.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return 130
        _print_json(result)
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Command line entry point for the gray routing / mirror middleware."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .engine.errors import ALL_CODES, GrayError
from .engine.jsonio import RationalEncoder, dump_file
from .engine.logging_utils import StructuredLogger
from .engine.replay import verify_replay
from .engine.session import Session, restore_session
from .engine.loader import load_rules, load_stream
from .engine.fixedpoint import scale_from_exponent


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, cls=RationalEncoder, ensure_ascii=False,
                     indent=2, sort_keys=True))


def _fail(error: GrayError, logger: StructuredLogger) -> int:
    logger.error_event("business_error", **error.to_dict())
    _print_json({"ok": False, "error": error.to_dict()})
    return 2


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--scale-exponent", required=True, type=int)
    parser.add_argument("--log", type=Path, default=Path("samples/logs/events.jsonl"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graymirror",
        description="Fixed point gray routing and traffic mirror middleware")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a routing session and print the report")
    _add_common(run)
    run.add_argument("--snapshot", type=Path)
    run.add_argument("--report", type=Path)

    snap = sub.add_parser("snapshot", help="run then persist a session snapshot")
    _add_common(snap)
    snap.add_argument("--snapshot", required=True, type=Path)

    restore = sub.add_parser("restore", help="restore a snapshot and verify replay")
    restore.add_argument("--rules", required=True, type=Path)
    restore.add_argument("--stream", required=True, type=Path)
    restore.add_argument("--snapshot", required=True, type=Path)
    restore.add_argument("--log", type=Path, default=Path("samples/logs/events.jsonl"))

    inject = sub.add_parser("inject", help="apply a deterministic fault injection")
    inject.add_argument(
        "--kind", required=True,
        choices=("weight_out_of_range", "timestamp_regression",
                 "snapshot_scale_missing"))
    inject.add_argument("--rules", type=Path, default=Path("samples/rules.json"))
    inject.add_argument("--stream", type=Path, default=Path("samples/stream.ndjson"))
    inject.add_argument("--snapshot", type=Path,
                        default=Path("samples/snapshots/session.json"))
    inject.add_argument("--scale-exponent", type=int, default=2)
    inject.add_argument("--log", type=Path, default=Path("samples/logs/events.jsonl"))
    inject.add_argument("--workdir", type=Path, default=Path("samples/injected"))

    return parser


def _run_session(args: argparse.Namespace, logger: StructuredLogger) -> Session:
    # E_GRID is raised here for an invalid exponent before any file is touched.
    scale_from_exponent(args.scale_exponent)
    rules = load_rules(args.rules)
    requests = load_stream(args.stream, rules)
    session = Session()
    session.replace(rules, args.scale_exponent, requests)
    logger.info_event(
        "session_run", rule_version=session.current.version,
        scale_exponent=args.scale_exponent,
        requests=session.current.stream_summary["count"])
    return session


def _report(view) -> dict[str, Any]:
    return {
        "ok": True,
        "rule_version": view.version,
        "stale": view.stale,
        "scale_exponent": view.scale_exponent,
        "scale_factor": view.scale,
        "stream_summary": view.stream_summary,
        "quantized_rules": view.quantized_report,
        "statistics": view.statistics,
        "decisions": view.decisions,
        "mirrors": view.mirrors,
        "change_ledger": view.change_ledger,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(args.log)
    try:
        if args.command in ("run", "snapshot"):
            session = _run_session(args, logger)
            if args.command == "snapshot":
                session.snapshot(args.snapshot)
                logger.info_event("snapshot_written", path=str(args.snapshot),
                                  rule_version=session.current.version)
            report = _report(session.current)
            if getattr(args, "snapshot", None) and args.command == "run":
                session.snapshot(args.snapshot)
            if getattr(args, "report", None):
                dump_file(args.report, report)
            _print_json(report)
            return 0

        if args.command == "restore":
            session, snapshot, _requests = restore_session(
                args.rules, args.stream, args.snapshot)
            verdict = verify_replay(session.current, snapshot)
            logger.info_event("replay_verified", **verdict)
            _print_json({"ok": True, "replay": verdict})
            return 0

        if args.command == "inject":
            from .injection import apply_injection
            result = apply_injection(args)
            logger.error_event("injection_reproduced", **result)
            _print_json({"ok": False, "injection": result})
            return 2

    except GrayError as error:
        if error.code not in ALL_CODES:  # defensive, keeps codes explicit
            error.code = E_VERSION
        return _fail(error, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())

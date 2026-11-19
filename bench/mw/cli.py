"""Command line interface for the middleware experiment workbench."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

from .config import canonical_dumps, load_events, load_spec
from .errors import (BenchError, OrderError, SnapshotError,
                     SpecError, StructuredLogger)
from .fault_inject import (inject_corrupt_event_file,
                           inject_snapshot_digest,
                           inject_timestamp_regression)
from .recovery import (continue_from_snapshot, unknown_after_crash,
                       verify_against_snapshot)
from .session import Session, restore_session

EXIT_CODES = {
    "E_SPEC": 2, "E_WATER": 3, "E_VIEW": 4, "E_ORDER": 5,
    "E_SNAPSHOT": 6, "E_REPLAY": 7,
}


def _emit(payload: Any) -> None:
    if is_dataclass(payload):
        payload = asdict(payload)
    sys.stdout.write(canonical_dumps(payload) + "\n")
    sys.stdout.flush()


def _default_paths(args: argparse.Namespace) -> dict[str, str]:
    base = args.workdir
    return {
        "ledger": args.ledger or os.path.join(base, "ledgers", "ledger.db"),
        "snapshot": args.snapshot or os.path.join(base, "snapshots",
                                                  "snapshot.json"),
        "log": args.log or os.path.join(base, "logs", "events.ndjson"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mwbench",
        description="Local deterministic partition/watermark workbench")
    parser.add_argument("--workdir", default="run",
                        help="runtime root for ledgers/snapshots/logs")
    parser.add_argument("--ledger")
    parser.add_argument("--snapshot")
    parser.add_argument("--log")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="validate spec + events and report")
    p_init.add_argument("--spec", required=True)
    p_init.add_argument("--events", required=True)

    p_ingest = sub.add_parser("ingest", help="load an NDJSON event batch")
    p_ingest.add_argument("--spec", required=True)
    p_ingest.add_argument("--events", required=True)
    p_ingest.add_argument("--drain", action="store_true")
    p_ingest.add_argument("--watermark", type=int)

    p_query = sub.add_parser("query", help="UI cache query with latency")
    p_query.add_argument("--spec", required=True)
    p_query.add_argument("--partition")
    p_query.add_argument("--inject-delay-ms", type=int, default=0)

    p_recompute = sub.add_parser(
        "recompute", help="authoritative replay and byte view comparison")
    p_recompute.add_argument("--spec", required=True)

    p_lat = sub.add_parser("latency", help="latency stats from query ledger")
    p_lat.add_argument("--spec", required=True)

    p_params = sub.add_parser(
        "set-params", help="replace parameters (new monotonic version)")
    p_params.add_argument("--spec", required=True)
    p_params.add_argument("--new-spec", required=True)

    p_snap = sub.add_parser("snapshot", help="write a session snapshot")
    p_snap.add_argument("--spec", required=True)

    p_restore = sub.add_parser("restore", help="restore from a snapshot")
    p_restore.add_argument("--spec", required=True)

    p_resume = sub.add_parser("resume", help="continue after restore")
    p_resume.add_argument("--spec", required=True)
    p_resume.add_argument("--events", required=True)

    p_crash = sub.add_parser(
        "crash-state", help="report unknown/reload state after a kill")
    p_crash.add_argument("--spec", required=True)

    p_inject = sub.add_parser("inject", help="run a fault injection")
    p_inject.add_argument("--spec", required=True)
    p_inject.add_argument("--events")
    p_inject.add_argument("--kind", required=True,
                          choices=["corrupt", "regression", "snapshot-digest"])
    p_inject.add_argument("--partition", default="p0")
    p_inject.add_argument("--seq", type=int, default=0)
    p_inject.add_argument("--ts-ms", type=int, default=-10_000)

    sub.add_parser("gui", help="launch the optional Tk dashboard")
    return parser


def _open_session(args: argparse.Namespace, spec_path: str,
                  logger: StructuredLogger, *,
                  event_file: str | None = None) -> Session:
    spec = load_spec(spec_path)
    paths = _default_paths(args)
    return Session(spec, paths["ledger"], logger, event_file=event_file)


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = _default_paths(args)
    os.makedirs(os.path.dirname(paths["ledger"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["snapshot"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["log"]), exist_ok=True)
    logger = StructuredLogger(paths["log"])
    try:
        return _dispatch(args, logger, paths)
    except BenchError as exc:
        record = logger.emit(exc.code, exc.message, **exc.details)
        _emit({"status": "error", "error": record})
        return EXIT_CODES.get(exc.code, 1)
    finally:
        logger.close()


def _dispatch(args: argparse.Namespace, logger: StructuredLogger,
              paths: dict[str, str]) -> int:
    if args.command == "init":
        spec = load_spec(args.spec)
        events = load_events(args.events, spec)
        _emit({"status": "ok", "partitions": list(spec.partition_keys),
               "events": len(events),
               "out_of_order_ms": spec.out_of_order_ms,
               "latency_budget_ms": spec.latency_budget_ms,
               "scale": spec.scale})
        return 0

    if args.command == "ingest":
        session = _open_session(args, args.spec, logger,
                                event_file=args.events)
        try:
            result = session.ingest_file(args.events)
            if args.watermark is not None:
                result["manual_watermark"] = session.advance_watermark(
                    args.watermark)
            if args.drain:
                result["drain"] = session.drain()
            result["cache_view"] = session.cache_view_bytes().decode("utf-8")
            _emit({"status": "ok", **result})
        finally:
            session.close()
        return 0

    if args.command == "query":
        session = _open_session(args, args.spec, logger)
        try:
            result = session.query(args.partition,
                                   inject_delay_ms=args.inject_delay_ms)
            _emit({"status": "ok", **asdict(result)})
        finally:
            session.close()
        return 0

    if args.command == "recompute":
        session = _open_session(args, args.spec, logger)
        try:
            verification = session.verify_views()
            _emit({"status": "ok", **verification,
                   "cache_view": session.cache_view_bytes().decode("utf-8"),
                   "authoritative_view":
                       session.authoritative_view_bytes().decode("utf-8")})
        finally:
            session.close()
        return 0

    if args.command == "latency":
        session = _open_session(args, args.spec, logger)
        try:
            _emit({"status": "ok", **session.latency_report()})
        finally:
            session.close()
        return 0

    if args.command == "set-params":
        session = _open_session(args, args.spec, logger)
        try:
            new_spec = load_spec(args.new_spec)
            _emit({"status": "ok", **session.replace_parameters(new_spec)})
        finally:
            session.close()
        return 0

    if args.command == "snapshot":
        session = _open_session(args, args.spec, logger)
        try:
            _emit({"status": "ok",
                   **session.save_snapshot(paths["snapshot"])})
        finally:
            session.close()
        return 0

    if args.command == "restore":
        session = restore_session(paths["snapshot"], paths["ledger"], logger)
        try:
            _emit({"status": "ok", "param_version": session.param_version,
                   "replacement_required": session.replacement_required})
        finally:
            session.close()
        return 0

    if args.command == "resume":
        session = restore_session(paths["snapshot"], paths["ledger"], logger)
        try:
            snapshot = session.snapshot_info
            before = verify_against_snapshot(session, snapshot)
            ingest = continue_from_snapshot(session, args.events, logger)
            session.drain()
            _emit({"status": "ok", "snapshot_check": before,
                   "continuation": ingest,
                   "final_view": session.cache_view_bytes().decode("utf-8")})
        finally:
            session.close()
        return 0

    if args.command == "crash-state":
        session = _open_session(args, args.spec, logger)
        try:
            _emit({"status": "ok", "exit_report": "not_generated",
                   "unknown": unknown_after_crash(session)})
        finally:
            session.close()
        return 0

    if args.command == "inject":
        return _run_injection(args, logger, paths)

    if args.command == "gui":
        try:
            from .gui import launch_gui
            launch_gui(paths["ledger"], paths["log"])
        except Exception as exc:  # environment without tkinter
            _emit({"status": "error", "code": "E_GUI_UNAVAILABLE",
                   "message": str(exc)})
            return 8
        return 0

    return 1  # argparse subparsers are required=True


def _run_injection(args: argparse.Namespace, logger: StructuredLogger,
                   paths: dict[str, str]) -> int:
    injected_dir = os.path.join(args.workdir, "injected")
    os.makedirs(injected_dir, exist_ok=True)
    if args.kind == "corrupt":
        target = os.path.join(injected_dir, "corrupt_events.ndjson")
        inject_corrupt_event_file(args.events, target)
        try:
            spec = load_spec(args.spec)
            load_events(target, spec)
        except SpecError as exc:
            record = logger.emit(exc.code, exc.message, **exc.details)
            _emit({"status": "error", "injection": "corrupt",
                   "error": record})
            return EXIT_CODES["E_SPEC"]
        _emit({"status": "unexpected", "injection": "corrupt"})
        return 1

    if args.kind == "regression":
        target = os.path.join(injected_dir, "regression_events.ndjson")
        inject_timestamp_regression(
            args.events, target, partition=args.partition,
            seq=args.seq, new_ts_ms=args.ts_ms)
        # Real resume path: restore the latest snapshot then continue.
        session = restore_session(paths["snapshot"], paths["ledger"], logger)
        try:
            continue_from_snapshot(session, target, logger)
            _emit({"status": "unexpected", "injection": "regression"})
        except OrderError as exc:
            # continue_from_snapshot already wrote the structured E_ORDER row.
            record = exc.details.get("log_record") or exc.to_record()
            _emit({"status": "error", "injection": "regression",
                   "error": record})
            return EXIT_CODES["E_ORDER"]
        finally:
            session.close()
        return 1

    # snapshot digest mismatch
    session = _open_session(args, args.spec, logger)
    try:
        session.save_snapshot(paths["snapshot"])
    finally:
        session.close()
    bad_snapshot = os.path.join(injected_dir, "tampered_snapshot.json")
    inject_snapshot_digest(paths["snapshot"], bad_snapshot)
    try:
        restore_session(bad_snapshot, paths["ledger"], logger, tampered=True)
    except SnapshotError as exc:
        record = logger.emit(exc.code, exc.message, **exc.details)
        _emit({"status": "error", "injection": "snapshot-digest",
               "error": record})
        return EXIT_CODES["E_SNAPSHOT"]
    _emit({"status": "unexpected", "injection": "snapshot-digest"})
    return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

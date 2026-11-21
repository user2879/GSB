"""Command line entry point for the cluster lease watermark component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import LeaseError
from .ledger import Ledger
from .loader import load_inputs
from .logging_utils import StructuredLogger
from .runtime import evolve_session, recover_session, run_session
from .snapshots import build_snapshot, open_snapshot_session

EXIT_CODES = {
    "E_SPEC": 2,
    "E_REF": 2,
    "E_TIMING": 2,
    "E_PORT": 3,
    "E_HANDSHAKE": 3,
    "E_REPLY": 3,
    "E_STATE": 3,
    "E_SNAPSHOT": 4,
    "E_REPLAY": 5,
    "E_ENV": 6,
}


def _add_inputs(parser: argparse.ArgumentParser, *, db_required: bool = True) -> None:
    parser.add_argument("--clusters", required=True, help="cluster declaration JSON")
    parser.add_argument("--holders", required=True, help="holder declaration JSON")
    parser.add_argument("--events", required=True, help="NDJSON lease event stream")
    parser.add_argument("--db", required=db_required, help="SQLite ledger path")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log", help="structured JSONL log path")
    parser.add_argument("--report", help="write the final JSON report to this path")
    parser.add_argument("--runtime-dir", default="build/runtime", help="service config directory")
    parser.add_argument("--startup-timeout-ms", type=int, default=5000)
    parser.add_argument("--handshake-timeout-ms", type=int, default=2000)


def _add_injection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inject",
        choices=("missing-ref", "timing", "snapshot"),
        default=argparse.SUPPRESS,
        help="controlled input-only fault injection",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cluster_lease")
    parser.add_argument(
        "--inject",
        choices=("missing-ref", "timing", "snapshot"),
        help="controlled input-only fault injection",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate all input files")
    _add_inputs(validate, db_required=False)
    _add_common(validate)
    _add_injection(validate)

    run = subparsers.add_parser("run", help="start services and process the event stream")
    _add_inputs(run)
    _add_common(run)
    _add_injection(run)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--clock-mode", choices=("logical", "hybrid", "real"), default="hybrid")
    run.add_argument("--snapshot-out", help="create a snapshot after the run")
    run.add_argument(
        "--crash-after-inflight-batch",
        type=int,
        help="acceptance-only forced termination after writing an uncommitted batch marker",
    )

    recover = subparsers.add_parser("recover", help="recover from committed records only")
    _add_inputs(recover)
    _add_common(recover)
    _add_injection(recover)
    recover.add_argument("--clock-mode", choices=("logical", "hybrid", "real"), default="logical")

    evolve = subparsers.add_parser("evolve", help="atomically replace state for changed parameters")
    _add_inputs(evolve)
    _add_common(evolve)
    _add_injection(evolve)

    snapshot_create = subparsers.add_parser("snapshot-create", help="create a self-contained snapshot")
    _add_inputs(snapshot_create)
    _add_common(snapshot_create)
    _add_injection(snapshot_create)
    snapshot_create.add_argument("--out", required=True)

    snapshot_open = subparsers.add_parser("snapshot-open", help="validate, restore and replay a snapshot")
    _add_inputs(snapshot_open)
    _add_common(snapshot_open)
    _add_injection(snapshot_open)
    snapshot_open.add_argument("--snapshot", required=True)

    gui = subparsers.add_parser("gui", help="start the read-only Tk diagnostics UI")
    gui.add_argument("--db", required=True)
    gui.add_argument("--log")

    inspect = subparsers.add_parser("inspect", help="emit current read-only ledger state")
    inspect.add_argument("--db", required=True)
    inspect.add_argument("--log")
    inspect.add_argument("--report")
    return parser


def _write_report(report: dict[str, Any], path: str | None) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_json(report: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def _ledger_inspect(db_path: str) -> dict[str, Any]:
    ledger = Ledger(db_path)
    try:
        clusters = []
        for version in ledger.cluster_versions():
            cluster_id = version["cluster_id"]
            clusters.append(
                {
                    "active_leases": ledger.current_active(cluster_id),
                    "cluster_id": cluster_id,
                    "decisions": ledger.current_decisions(cluster_id),
                    "ownership_intervals": ledger.current_ownership_intervals(cluster_id),
                    "parameter_version": version["parameter_version"],
                    "parameters": version,
                    "progress": ledger.progress_for(cluster_id),
                    "watermarks": ledger.current_watermarks(cluster_id),
                }
            )
        return {
            "batch_runs": ledger.batch_runs(),
            "clusters": clusters,
            "content_digest": ledger.content_digest(),
            "format_version": int(ledger.get_meta("ledger_format_version", "1") or 1),
            "ok": True,
            "snapshots": ledger.snapshot_catalog(),
            "stream_cursor": ledger.stream_cursor(),
        }
    finally:
        ledger.close()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command
    log_path = args.log or f"logs/{command}.jsonl"
    logger = StructuredLogger(log_path)
    try:
        if command == "gui":
            try:
                from .gui import run_gui
            except ImportError as exc:
                raise LeaseError(
                    "E_ENV",
                    f"tkinter is not available in this Python installation: {exc}",
                    field="tkinter",
                ) from None

            run_gui(args.db)
            return 0
        if command == "inspect":
            report = _ledger_inspect(args.db)
            logger.log("LEDGER_INSPECT", "read-only ledger state emitted", content_digest=report["content_digest"])
            _write_report(report, args.report)
            _print_json(report)
            return 0

        logger.log(
            "INPUT_LOAD_STARTED",
            "loading and validating local declarations",
            clusters=args.clusters,
            events=args.events,
            holders=args.holders,
            injection=args.inject,
        )
        bundle = load_inputs(
            args.clusters,
            args.holders,
            args.events,
            injection=args.inject,
        )
        logger.log(
            "INPUT_LOAD_OK",
            "input validation passed",
            cluster_count=len(bundle.clusters),
            event_count=len(bundle.events),
            event_stream_digest=bundle.event_stream_digest(),
            holder_count=len(bundle.holders),
        )
        if command == "validate":
            report = {
                "cluster_count": len(bundle.clusters),
                "clusters": [cluster.to_json() for cluster in bundle.clusters],
                "code": "OK",
                "event_count": len(bundle.events),
                "event_stream_digest": bundle.event_stream_digest(),
                "holder_count": len(bundle.holders),
                "ok": True,
            }
        elif command == "run":
            if args.inject == "snapshot":
                raise LeaseError(
                    "E_SPEC",
                    "--inject snapshot is only valid with snapshot-open",
                    field="injection",
                )
            report = run_session(
                bundle,
                args.db,
                runtime_dir=args.runtime_dir,
                logger=logger,
                batch_size=args.batch_size,
                clock_mode=args.clock_mode,
                startup_timeout_ms=args.startup_timeout_ms,
                handshake_timeout_ms=args.handshake_timeout_ms,
                crash_after_inflight_batch=args.crash_after_inflight_batch,
            )
            if args.snapshot_out:
                ledger = Ledger(args.db)
                try:
                    snapshot = build_snapshot(ledger, bundle, args.snapshot_out)
                finally:
                    ledger.close()
                report["snapshot"] = {
                    "content_digest": snapshot["content_digest"],
                    "path": args.snapshot_out,
                    "snapshot_id": snapshot["snapshot_id"],
                }
        elif command == "recover":
            report = recover_session(
                bundle,
                args.db,
                runtime_dir=args.runtime_dir,
                logger=logger,
                clock_mode=args.clock_mode,
                startup_timeout_ms=args.startup_timeout_ms,
                handshake_timeout_ms=args.handshake_timeout_ms,
            )
        elif command == "evolve":
            report = evolve_session(
                bundle,
                args.db,
                runtime_dir=args.runtime_dir,
                logger=logger,
                startup_timeout_ms=args.startup_timeout_ms,
                handshake_timeout_ms=args.handshake_timeout_ms,
            )
        elif command == "snapshot-create":
            ledger = Ledger(args.db)
            try:
                snapshot = build_snapshot(ledger, bundle, args.out)
            finally:
                ledger.close()
            report = {
                "code": "OK",
                "content_digest": snapshot["content_digest"],
                "event_stream_digest": snapshot["event_stream_digest"],
                "format_version": snapshot["format_version"],
                "ok": True,
                "path": args.out,
                "snapshot_id": snapshot["snapshot_id"],
            }
        elif command == "snapshot-open":
            report = open_snapshot_session(
                bundle,
                args.db,
                args.snapshot,
                runtime_dir=args.runtime_dir,
                logger=logger,
                injection=args.inject,
                startup_timeout_ms=args.startup_timeout_ms,
                handshake_timeout_ms=args.handshake_timeout_ms,
            )
        else:
            raise LeaseError("E_SPEC", f"unknown command: {command}", field="command")
        logger.log(
            "COMMAND_COMPLETED",
            "command completed",
            command=command,
            code=report.get("code", "OK"),
        )
        _write_report(report, getattr(args, "report", None))
        _print_json(report)
        return EXIT_CODES.get(str(report.get("code")), 0)
    except LeaseError as exc:
        logger.log(
            exc.code,
            exc.message,
            field=exc.field,
            details=exc.details,
            command=command,
        )
        report = exc.to_dict()
        _write_report(report, getattr(args, "report", None))
        _print_json(report)
        return EXIT_CODES.get(exc.code, 2)
    except Exception as exc:
        logger.log("E_INTERNAL", str(exc), command=command)
        report = {
            "code": "E_INTERNAL",
            "message": str(exc),
            "ok": False,
        }
        _write_report(report, getattr(args, "report", None))
        _print_json(report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

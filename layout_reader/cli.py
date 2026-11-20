"""Command-line entry points for the layout workbench."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigError, load_config
from .faults import load_fault_registry
from .pipeline import run_pipeline
from .store import SnapshotStore


def _analyze(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        faults = load_fault_registry(args.faults)
    except (ConfigError, OSError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    stop_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
        except (ValueError, OSError):
            pass

    workers = 1 if args.serial else args.workers
    try:
        report = run_pipeline(
            input_dir=args.input,
            config=config,
            fault_registry=faults,
            state_dir=args.state,
            truth_path=args.truth,
            workers=workers,
            stop_event=stop_event,
            report_path=args.report,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass

    summary = report["summary"]
    print(
        "status={status} snapshot=v{version} pages={pages} ok={ok} failed={failed} "
        "reused_stages={reused} recomputed_stages={recomputed} injection_hits={hits} "
        "truth_passed={truth}".format(
            status=report["status"],
            version=report["snapshot"]["committed_version"],
            pages=summary["input_page_count"],
            ok=summary["completed_page_count"],
            failed=summary["failed_page_count"],
            reused=summary["reused_stage_count"],
            recomputed=summary["recomputed_stage_count"],
            hits=summary["injection_hit_count"],
            truth=summary["truth_validation_passed"],
        )
    )
    if report["status"] == "interrupted":
        pending = report.get("pending_pages", [])
        print(f"graceful_stop=1 pending_pages={len(pending)}")
    if args.report:
        print(f"report={args.report}")
    return 0


def _status(args: argparse.Namespace) -> int:
    status = SnapshotStore(args.state).status()
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layout-workbench",
        description="Deterministic local layout reading-order workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze PNG pages")
    analyze.add_argument("--input", default="samples", help="directory containing PNG pages")
    analyze.add_argument(
        "--config", default="configs/default.json", help="layout analysis config JSON"
    )
    analyze.add_argument(
        "--faults", default=None, help="fault-injection list JSON (omit for none)"
    )
    analyze.add_argument("--state", default="runtime", help="snapshot state directory")
    analyze.add_argument("--report", default="out/report.json", help="diagnostic report path")
    analyze.add_argument("--truth", default=None, help="explicit truth JSON path")
    analyze.add_argument("--workers", type=int, default=None, help="override thread workers")
    analyze.add_argument(
        "--serial", action="store_true", help="force one worker; equivalent to --workers 1"
    )
    analyze.set_defaults(func=_analyze)

    status = subparsers.add_parser("status", help="show the committed snapshot")
    status.add_argument("--state", default="runtime", help="snapshot state directory")
    status.set_defaults(func=_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze" and args.serial and args.workers not in (None, 1):
        parser.error("--serial and --workers > 1 are mutually exclusive")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


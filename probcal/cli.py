from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .service import edit_service, run_service, verify_service
from .util import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probcal",
        description="Deterministic tabular probability calibration and incremental reuse service",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run or resume the four-stage pipeline")
    _common_arguments(run)
    run.set_defaults(handler=_handle_run)
    edit = subparsers.add_parser("edit", help="apply a controlled edit and incrementally rerun")
    _common_arguments(edit)
    edit.add_argument("--edit", required=True, type=Path, help="YAML edit command file")
    edit.set_defaults(handler=_handle_edit)
    verify = subparsers.add_parser("verify", help="inspect the current manifest and snapshot")
    _common_arguments(verify)
    verify.set_defaults(handler=_handle_verify)
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--state", type=Path, default=Path(".probcal_state"))
    parser.add_argument("--expect-version", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON report here")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    stop_event = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), request_stop)
            except (OSError, ValueError):
                pass
    try:
        report = args.handler(args, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        return 0
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "business_code": "E_RUNTIME",
            "message": f"{type(exc).__name__}: {exc}",
            "failures": [
                {
                    "code": "E_RUNTIME",
                    "stage": "runtime",
                    "field": "uncaught",
                    "message": str(exc),
                }
            ],
        }
    if args.output is not None:
        atomic_write_json(args.output, report)
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()
    if report.get("status") == "ok" or report.get("status") == "interrupted":
        return 0
    return 2


def _handle_run(args: argparse.Namespace, stop_event: threading.Event) -> dict[str, Any]:
    return run_service(
        args.config,
        args.state,
        expect_version=args.expect_version,
        stop_event=stop_event,
    )


def _handle_edit(args: argparse.Namespace, stop_event: threading.Event) -> dict[str, Any]:
    return edit_service(
        args.config,
        args.state,
        args.edit,
        expect_version=args.expect_version,
        stop_event=stop_event,
    )


def _handle_verify(args: argparse.Namespace, stop_event: threading.Event) -> dict[str, Any]:
    return verify_service(args.state)

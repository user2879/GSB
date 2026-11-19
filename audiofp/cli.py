"""Command-line interface with SIGINT/SIGTERM cancellation and JSON errors."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from .demo import main as demo_main
from .engine import run_task
from .errors import BusinessError
from .runtime import CancelToken
from .task import load_task


def _emit(document: dict[str, Any]) -> None:
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))


def _exit_code(error: BusinessError) -> int:
    if error.code == "E_INTERRUPTED":
        return 130
    if error.code in {"E_SCHEMA", "E_JSON", "E_RATE", "E_BAND"}:
        return 2
    return 1


def _run_command(task_path: Path, *, quiet: bool) -> int:
    token = CancelToken()

    def handle_signal(signum: int, _frame: Any) -> None:
        token.cancel(f"received signal {signum}")

    previous_int = signal.signal(signal.SIGINT, handle_signal)
    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        task = load_task(task_path)
        report = run_task(task, cancel_token=token)
        if not quiet:
            _emit(
                {
                    "business_code": report["business_code"],
                    "status": report["status"],
                    "report": str(task.report_path),
                    "results": str(task.results_path),
                    "pairs": len(report["pairs"]),
                    "fragments": sum(len(pair["fragments"]) for pair in report["pairs"]),
                }
            )
        return 0
    except BusinessError as error:
        _emit(error.to_dict())
        return _exit_code(error)
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audiofp", description="Local audio fingerprint retrieval engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one retrieval task")
    run_parser.add_argument("task", type=Path)
    run_parser.add_argument("--quiet", action="store_true")
    demo_parser = subparsers.add_parser("demo", help="generate the deterministic corpus and run the demo")
    demo_parser.add_argument("--workspace", default=".")
    demo_parser.add_argument("--no-clean", action="store_true")
    demo_parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args.task, quiet=args.quiet)
    if args.command == "demo":
        demo_argv = ["--workspace", args.workspace, "--runs", str(args.runs)]
        if args.no_clean:
            demo_argv.append("--no-clean")
        return demo_main(demo_argv)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


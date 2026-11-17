"""Command-line entry points for batch runs and the interactive editor."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Any, Sequence

from .canonical import atomic_write_json
from .data import read_structured_file
from .pipeline import run_pipeline


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_code": report.get("exit_code"),
        "run_state": report.get("run_state"),
        "snapshot_version": report.get("snapshot_version"),
        "resumed": report.get("resumed", False),
        "resumed_stages": report.get("resumed_stages", []),
        "tasks": [
            {
                "name": task.get("name"),
                "accuracy": task.get("metrics", {}).get("accuracy"),
                "macro_f1": task.get("metrics", {}).get("macro_f1"),
                "cache": task.get("cache"),
            }
            for task in report.get("tasks", [])
        ],
        "incremental": report.get("incremental"),
        "cache": report.get("cache"),
        "failures": [
            {
                "task_name": failure.get("task_name"),
                "stage": failure.get("stage"),
                "code": failure.get("code"),
                "exception_type": failure.get("exception_type"),
            }
            for failure in report.get("failures", [])
        ],
        "injection_hits": report.get("injection_hits", []),
    }


def print_summary(report: dict[str, Any]) -> None:
    print(json.dumps(_summary(report), ensure_ascii=False, sort_keys=True))


class _SignalGuard:
    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "_SignalGuard":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def _handle(self, signum: int, frame: Any) -> None:
        self.stop_event.set()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def _load_document(path: str | None) -> Any:
    return read_structured_file(path) if path is not None else None


def _run_command(args: argparse.Namespace) -> int:
    try:
        tasks = read_structured_file(args.tasks)
        features = read_structured_file(args.features)
        encoder = read_structured_file(args.encoder)
        adapter_config = read_structured_file(args.config)
        injections = _load_document(args.injections)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "run_state": "failed",
            "exit_code": 2,
            "tasks": [],
            "failures": [
                {
                    "task_name": None,
                    "stage": "input",
                    "code": "E_INPUT_INVALID",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "injection_ids": [],
                }
            ],
        }
        if args.report:
            atomic_write_json(args.report, report)
        print_summary(report)
        return 2
    stop_event = threading.Event()
    with _SignalGuard(stop_event):
        result = run_pipeline(
            tasks,
            features,
            encoder,
            adapter_config,
            state_directory=args.state_dir,
            injection_document=injections,
            report_path=args.report,
            stop_event=stop_event,
            expected_version=args.expected_version,
            expected_config_hash=args.expected_config_hash,
            require_resume=args.resume,
            force_full=args.force_full,
        )
    print_summary(dict(result.report))
    return result.exit_code


def _repl_command(args: argparse.Namespace) -> int:
    from .repl import run_repl

    return run_repl(args)


def _verify_command(args: argparse.Namespace) -> int:
    from .verification import run_acceptance_verification

    result = run_acceptance_verification(
        examples_directory=args.examples,
        output_directory=args.output,
    )
    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m transfer_pipeline",
        description="Deterministic incremental cross-task adapter pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run or incrementally update once")
    run_parser.add_argument("--tasks", required=True, help="task description JSON/YAML")
    run_parser.add_argument("--features", required=True, help="synthetic sample library")
    run_parser.add_argument("--encoder", required=True, help="shared encoder package")
    run_parser.add_argument("--config", required=True, help="adapter configuration")
    run_parser.add_argument("--injections", help="optional exception injection config")
    run_parser.add_argument("--state-dir", required=True, help="atomic snapshot directory")
    run_parser.add_argument("--report", help="write the complete JSON report here")
    run_parser.add_argument("--expected-version", type=int)
    run_parser.add_argument("--expected-config-hash")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="require a matching interrupted snapshot (else E_CONFIG_MISMATCH)",
    )
    run_parser.add_argument(
        "--force-full",
        action="store_true",
        help="ignore content caches and train every valid task from scratch",
    )
    run_parser.set_defaults(handler=_run_command)

    repl_parser = subparsers.add_parser(
        "repl", help="interactive add/delete/edit commands with automatic incremental runs"
    )
    repl_parser.add_argument("--tasks", required=True)
    repl_parser.add_argument("--features", required=True)
    repl_parser.add_argument("--encoder", required=True)
    repl_parser.add_argument("--config", required=True)
    repl_parser.add_argument("--injections")
    repl_parser.add_argument("--state-dir", required=True)
    repl_parser.add_argument("--report", required=True)
    repl_parser.add_argument(
        "--manual",
        action="store_true",
        help="do not run automatically after a mutating command",
    )
    repl_parser.set_defaults(handler=_repl_command)

    verify_parser = subparsers.add_parser(
        "verify", help="run the shipped incremental-versus-full acceptance sequence"
    )
    verify_parser.add_argument("--examples", default="examples")
    verify_parser.add_argument("--output", default="artifacts")
    verify_parser.set_defaults(handler=_verify_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())

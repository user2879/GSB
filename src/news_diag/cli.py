from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path


_STOP_EVENT = threading.Event()


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def handle_signal(signum: int, _frame: object) -> None:
        stop_event.set()
        print(
            f"received signal {signum}; finishing active documents and checkpointing state",
            file=sys.stderr,
        )

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        candidate = getattr(signal, signal_name, None)
        if candidate is None:
            continue
        try:
            signal.signal(candidate, handle_signal)
        except (OSError, ValueError):
            continue


# Install before importing the numeric dependencies so a cooperative interrupt during start-up
# is still converted into checkpoint-and-exit semantics.
_install_signal_handlers(_STOP_EVENT)

from .engine import RunnerOptions, run_workbench  # noqa: E402
from .io_utils import atomic_write_text, json_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chinese news summary and style-rewrite diagnostics workbench"
    )
    parser.add_argument("--news", required=True, type=Path, help="news JSON/JSONL file or directory")
    parser.add_argument("--config", required=True, type=Path, help="summary/rewrite YAML config")
    parser.add_argument("--injections", type=Path, default=None, help="perturbation JSON file")
    parser.add_argument("--report", type=Path, default=None, help="write deterministic JSON report")
    parser.add_argument("--stdout", action="store_true", help="also print report JSON to stdout")
    parser.add_argument("--state-dir", type=Path, default=None, help="checkpoint/progress directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed documents from --state-dir only",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="run documents on one thread while preserving the same stable ordering",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.state_dir is None:
        parser.error("--resume requires --state-dir")

    options = RunnerOptions(
        news_path=args.news,
        config_path=args.config,
        injections_path=args.injections,
        serial=args.serial,
        state_dir=args.state_dir,
        resume=args.resume,
        stop_event=_STOP_EVENT,
    )
    try:
        result = run_workbench(options)
    except KeyboardInterrupt:
        # The handler normally turns SIGINT into a cooperative stop. This path only remains for
        # environments where the default handler wins during interpreter start-up.
        print("interrupted before a report could be produced; persisted state is authoritative", file=sys.stderr)
        return 0

    text = json_text(result.report)
    if args.report is not None:
        atomic_write_text(args.report, text)
    if args.stdout or args.report is None:
        sys.stdout.write(text)
    if result.interrupted:
        print(
            f"graceful stop: {result.completed_document_count} documents completed; "
            f"{len(result.pending_document_ids)} pending",
            file=sys.stderr,
        )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

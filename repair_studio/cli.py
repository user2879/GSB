from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import (
    RunController,
    install_graceful_signal_handlers,
    recover_output,
    restore_signal_handlers,
)
from .samplegen import generate_default_samples


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def command_generate(arguments: argparse.Namespace) -> int:
    generated = generate_default_samples(arguments.root)
    _print_json({"generated": generated})
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    controller = RunController(arguments.samples, arguments.output, arguments.mode)
    previous_handlers = install_graceful_signal_handlers(controller.stop_event)
    try:
        report = controller.run()
    finally:
        restore_signal_handlers(previous_handlers)
    _print_json(
        {
            "run_id": report["run_id"],
            "mode": report["mode"],
            "report_path": report["report_path"],
            "summary": report["summary"],
            "failures": report["failures"],
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "status": sample["status"],
                    "completed_command_cursor": sample.get("completed_command_cursor"),
                    "final_png_path": sample.get("final_png_path"),
                }
                for sample in report["samples"]
            ],
            "graceful_exit": controller.stop_event.is_set(),
        }
    )
    return 0


def command_recover(arguments: argparse.Namespace) -> int:
    result = recover_output(arguments.output)
    _print_json(result)
    return 0 if result["status"] == "recovered" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repair-studio",
        description="Deterministic local image restoration and style transfer studio",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate deterministic sample data")
    generate_parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "samples")
    generate_parser.set_defaults(function=command_generate)

    run_parser = subparsers.add_parser("run", help="run a sample command stream")
    run_parser.add_argument("--samples", type=Path, default=PROJECT_ROOT / "samples")
    run_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    run_parser.add_argument("--mode", choices=["interactive", "sync"], default="interactive")
    run_parser.set_defaults(function=command_run)

    recover_parser = subparsers.add_parser("recover", help="recover only complete atomic snapshots")
    recover_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    recover_parser.set_defaults(function=command_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.function(arguments))


if __name__ == "__main__":
    sys.exit(main())


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .pipeline import Pipeline
from .runtime import ShutdownController


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peft-lora",
        description="Deterministic synthetic PEFT low-rank adapter pipeline",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="run the staged PEFT workflow")
    run_parser.add_argument("--samples", required=True, help="sample directory")
    run_parser.add_argument("--workspace", default="work", help="workspace directory")
    run_parser.add_argument(
        "--base-dir",
        default=None,
        help="override the base package path from peft_config.yaml",
    )
    inspect_parser = subparsers.add_parser("inspect", help="print CURRENT snapshot metadata")
    inspect_parser.add_argument("--workspace", default="work", help="workspace directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 2
    if arguments.command == "inspect":
        current = Path(arguments.workspace) / "snapshot" / "CURRENT.json"
        if not current.is_file():
            print(json.dumps({"status": "absent", "current": None}, sort_keys=True))
            return 1
        with open(current, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        print(json.dumps(value, sort_keys=True, ensure_ascii=False))
        return 0

    controller = ShutdownController()
    controller.install()
    try:
        pipeline = Pipeline(
            arguments.samples,
            arguments.workspace,
            base_directory=arguments.base_dir,
            controller=controller,
        )
        result = pipeline.run()
    finally:
        controller.restore()

    summary = {
        "status": result.status,
        "exit_code": result.exit_code,
        "report": str(result.report_path),
        "snapshot_version": (result.report.get("snapshot") or {}).get("version"),
        "recovery": result.report.get("recovery"),
        "business_codes": result.report.get("business_codes", []),
        "failures": result.report.get("failures", []),
        "reuse": result.report.get("reuse", []),
        "retrain": result.report.get("retrain", []),
        "training_executed": result.report.get("training_executed", []),
        "shutdown": result.report.get("shutdown"),
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


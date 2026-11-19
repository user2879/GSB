"""Command-line interface for deterministic evidence-chain runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import E_CALIB_BUDGET_PENDING, E_CONFIG_MISMATCH
from .engine import EnginePaths, run_engine
from .errors import VoicecalError
from .util import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicecal", description="Deterministic acoustic evidence-chain calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run or resume calibration")
    run.add_argument("--sample-dir", type=Path, default=Path("samples"))
    run.add_argument("--dialogue", type=Path, default=Path("samples/dialogue.json"))
    run.add_argument("--config", type=Path, default=Path("samples/calibration.json"))
    run.add_argument("--edits", type=Path, default=Path("samples/edits.json"))
    run.add_argument("--injections", type=Path, default=None)
    run.add_argument("--snapshot-dir", type=Path, default=Path("snapshots/demo"))
    run.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    run.add_argument("--fresh", action="store_true", help="ignore existing snapshot and run from a clean state")
    run.add_argument("--one-batch", action="store_true", help="stop after the next edit batch")
    return parser


def run_command(args: argparse.Namespace) -> int:
    paths = EnginePaths(
        sample_dir=args.sample_dir,
        dialogue_path=args.dialogue,
        config_path=args.config,
        edits_path=args.edits,
        injections_path=args.injections,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
    )
    try:
        report = run_engine(paths, fresh=args.fresh, process_all_batches=not args.one_batch)
    except VoicecalError as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", **exc.as_dict()}
        write_json(args.output_dir / "error.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if E_CALIB_BUDGET_PENDING in report.get("business_codes", []):
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

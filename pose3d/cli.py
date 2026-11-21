"""Command line interface for sample generation and solving."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError
from .pipeline import run_pipeline
from .samplegen import generate_sample


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _cmd_generate(args: argparse.Namespace) -> int:
    result = generate_sample(args.out, overwrite=not args.keep)
    _print_json(result)
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    input_dir = Path(args.input) if args.input else config_path.parent
    try:
        report = run_pipeline(
            input_dir=input_dir,
            config_path=config_path,
            injections_path=args.injections,
            snapshot_dir=args.snapshot_dir,
            output_path=args.output,
            fresh=args.fresh,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "report": str(Path(args.output).resolve()) if args.output else None,
        "input_version": report["input"]["input_version"],
        "snapshot_version": report["snapshot"]["global_version"],
        "snapshot_decision": report["snapshot"]["decision"],
        "recovery_decision": report["snapshot"]["recovery_decision"],
        "summary": report["summary"],
        "pending_queue": report["pending_queue"],
        "injection_hits": [
            item["id"] for item in report["injections"] if item.get("hit")
        ],
        "cache": report["cache"],
    }
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose3d",
        description="Deterministic multi-view 3D keypoint solver",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-sample", help="generate the fixed-seed local sample"
    )
    generate.add_argument("--out", required=True, help="sample output directory")
    generate.add_argument(
        "--keep",
        action="store_true",
        help="do not delete an existing output directory",
    )
    generate.set_defaults(func=_cmd_generate)

    solve = subparsers.add_parser("solve", help="run the solver and write a JSON report")
    solve.add_argument("--input", help="sample input directory (default: config parent)")
    solve.add_argument("--config", required=True, help="solver configuration JSON")
    solve.add_argument(
        "--injections", help="optional independent fault-injection JSON"
    )
    solve.add_argument(
        "--snapshot-dir",
        default="state/snapshots",
        help="snapshot/cache root (default: state/snapshots)",
    )
    solve.add_argument("--output", help="JSON report output path")
    solve.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing stage cache and perform a full rebuild",
    )
    solve.set_defaults(func=_cmd_solve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

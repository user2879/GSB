from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .conversion import run_conversion
from .edits import apply_operation, copy_dataset
from .engine import PipelineOptions, SimulatedTermination, run_pipeline
from .errors import BusinessError
from .generator import generate_dataset
from .injection import apply_injections
from .util import atomic_write_json, read_json
from .validation import run_validation


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="candidate_exchange",
        description="Offline candidate-box exchange, matching, incremental and snapshot validator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate the deterministic sample dataset")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=20261122)
    generate.add_argument("--force", action="store_true")

    inject = subparsers.add_parser("inject", help="apply a real fault-injection configuration")
    inject.add_argument("--data", required=True)
    inject.add_argument("--out", required=True)
    inject.add_argument("--injections", required=True)
    inject.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="run detection, matching, conversion and reporting")
    run.add_argument("--data", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--snapshots", required=True)
    run.add_argument("--active-version", choices=["v1", "v2"])
    run.add_argument("--workers", type=int)
    run.add_argument("--snapshot-interval", type=int)
    run.add_argument("--crash-after-stage", choices=["detection", "matching", "serialization"])
    run.add_argument("--force-full", action="store_true")
    run.add_argument("--injections")

    convert = subparsers.add_parser("convert", help="convert both versions without matching")
    convert.add_argument("--data", required=True)
    convert.add_argument("--out", required=True)

    edit = subparsers.add_parser("edit", help="copy a dataset and apply edit_sequence.json operations")
    edit.add_argument("--data", required=True)
    edit.add_argument("--out", required=True)
    edit.add_argument("--operations")
    edit.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="run the complete local acceptance suite")
    validate.add_argument("--data", required=True)
    validate.add_argument("--out", default="reports")
    validate.add_argument("--seed", type=int, default=20261122)
    validate.add_argument("--keep-work", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            _print_json(generate_dataset(args.out, seed=args.seed, force=args.force))
            return 0
        if args.command == "inject":
            _print_json(
                apply_injections(
                    args.data,
                    args.out,
                    args.injections,
                    force=args.force,
                )
            )
            return 0
        if args.command == "run":
            report = run_pipeline(
                PipelineOptions(
                    data_dir=Path(args.data),
                    output_dir=Path(args.out),
                    snapshot_root=Path(args.snapshots),
                    active_version=args.active_version,
                    workers=args.workers,
                    snapshot_commit_interval=args.snapshot_interval,
                    crash_after_stage=args.crash_after_stage,
                    force_full=args.force_full,
                    injections_path=Path(args.injections) if args.injections else None,
                )
            )
            _print_json(
                {
                    "status": report["status"],
                    "summary": report["summary"],
                    "report_path": report["report_path"],
                    "recovery_point_version": report["snapshot"]["recovery_point_version"],
                }
            )
            return 0
        if args.command == "convert":
            from .config import load_detection_params, load_match_config
            from .v1 import parse_v1
            from .v2 import parse_v2

            data = Path(args.data)
            conversion = run_conversion(
                v1=parse_v1(data / "candidates_v1.txt"),
                v2=parse_v2(data / "candidates_v2.json"),
                detection=load_detection_params(data),
                match=load_match_config(data),
                output_dir=args.out,
            )
            _print_json(conversion)
            return 0
        if args.command == "edit":
            copy_dataset(args.data, args.out, force=args.force)
            operations_path = Path(args.operations) if args.operations else Path(args.out) / "edit_sequence.json"
            payload = read_json(operations_path)
            outcomes = []
            for operation in payload.get("operations", []):
                outcomes.append(
                    {
                        "operation": operation,
                        "result": apply_operation(args.out, operation),
                    }
                )
            _print_json({"output_dir": str(Path(args.out).resolve()), "outcomes": outcomes})
            return 0
        if args.command == "validate":
            result = run_validation(
                Path(args.data),
                Path(args.out),
                seed=args.seed,
                keep_work=args.keep_work,
            )
            _print_json(result["summary"])
            return 0 if result["summary"]["all_checks_passed"] else 1
    except SimulatedTermination as exc:
        print(
            f"SIMULATED_TERMINATION after {exc.stage}; recovery version={exc.version}",
            file=sys.stderr,
        )
        return 75
    except BusinessError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"E_IO: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"E_UNEXPECTED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 3
    return 0

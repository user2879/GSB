from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .constants import APP_NAME
from .loader import InputPaths, LoadedInputs, load_inputs
from .reporting import write_report
from .runner import run_full, run_incremental, run_verification, write_validation_report


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="data", help="directory containing the five input files")
    parser.add_argument("--seeds", help="override seed JSONL path")
    parser.add_argument("--lexicon", help="override lexicon/tokenizer JSON path")
    parser.add_argument("--scoring", help="override scoring parameter JSON path")
    parser.add_argument("--config", help="override construction config JSON path")
    parser.add_argument("--plan", help="override batch plan JSON path")


def _load(args: argparse.Namespace) -> LoadedInputs:
    overrides = {
        key: getattr(args, key)
        for key in ("seeds", "lexicon", "scoring", "config", "plan")
        if getattr(args, key)
    }
    return load_inputs(args.input_dir, overrides=overrides)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _fatal_load_exit(inputs: LoadedInputs, output_dir: str | Path, mode: str) -> int:
    report = write_validation_report(inputs, output_dir)
    _print_json(
        {
            "ok": False,
            "mode": mode,
            "failure_codes": report["failure_codes"],
            "report": str(Path(output_dir).resolve() / "reports" / "validation.json"),
        }
    )
    return 2


def _validate_command(args: argparse.Namespace) -> int:
    inputs = _load(args)
    output = Path(args.output_dir).resolve()
    report = write_validation_report(inputs, output)
    valid = bool(report["valid"])
    _print_json(
        {
            "ok": valid,
            "fatal": not valid,
            "failure_codes": report["failure_codes"],
            "counts": {
                "seeds": len(inputs.seeds_by_id),
                "tasks": len(inputs.config.get("task_types", {})),
                "batches": len(inputs.plan.get("batches", [])),
            },
            "report": str(output / "reports" / "validation.json"),
        }
    )
    return 0 if valid else 1


def _run_command(args: argparse.Namespace) -> int:
    inputs = _load(args)
    output = Path(args.output_dir).resolve()
    if inputs.failures.fatal:
        return _fatal_load_exit(inputs, output, args.mode)
    if args.mode == "incremental":
        report, code = run_incremental(inputs=inputs, output_dir=output, reset=args.reset)
    else:
        report, code = run_full(inputs=inputs, output_dir=output, reset=args.reset)
    _print_json(
        {
            "ok": code == 0,
            "mode": args.mode,
            "candidate_count": len(report.get("candidates", [])),
            "retained_count": len(report.get("retained_candidate_ids", [])),
            "failure_codes": report.get("failure_business_codes", report.get("failure_codes", [])),
            "snapshot_version": (report.get("snapshot") or {}).get("global_version"),
            "report": str(output / "reports" / "latest.json"),
        }
    )
    return code


def _verify_command(args: argparse.Namespace) -> int:
    inputs = _load(args)
    output = Path(args.output_dir).resolve()
    if inputs.failures.fatal:
        return _fatal_load_exit(inputs, output, "verify")
    result, code = run_verification(inputs=inputs, output_dir=output)
    _print_json(result)
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="Deterministic local instruction-tuning data constructor",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate all local inputs")
    _add_input_arguments(validate_parser)
    validate_parser.add_argument("--output-dir", default="out")
    validate_parser.set_defaults(func=_validate_command)

    run_parser = subparsers.add_parser("run", help="construct data in incremental or full mode")
    _add_input_arguments(run_parser)
    run_parser.add_argument("--output-dir", default="out")
    run_parser.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    run_parser.add_argument("--reset", action="store_true", help="discard snapshot, cache, and reports first")
    run_parser.set_defaults(func=_run_command)

    verify_parser = subparsers.add_parser("verify", help="compare incremental, full, and repeated full runs")
    _add_input_arguments(verify_parser)
    verify_parser.add_argument("--output-dir", default="out/verification")
    verify_parser.set_defaults(func=_verify_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

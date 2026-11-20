"""Command-line interface for generation, runs, and the reproducible demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .errors import PipelineConfigMismatchError, PipelineError, SimulatedCrash
from .pipeline import run_pipeline
from .samplegen import generate_sample
from .workflow import run_demo, run_edit_sequence


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m transfer_pipeline",
        description="Deterministic incremental cross-task transfer-learning pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate fixed-seed local sample inputs")
    generate.add_argument("--out", required=True, help="sample output directory")
    generate.add_argument("--seed", type=int, default=20261120)
    generate.add_argument("--keep-existing", action="store_true")

    run = subparsers.add_parser("run", help="run one input state")
    run.add_argument("--input", required=True, help="sample input directory")
    run.add_argument("--workspace", required=True, help="snapshot workspace")
    run.add_argument("--report", help="report JSON path")
    run.add_argument("--injections", help="optional independent injection config")
    run.add_argument("--full", action="store_true", help="clean rebuild instead of resume")
    run.add_argument("--fresh", action="store_true", help="clear snapshots before running")
    run.add_argument("--allow-config-migration", action="store_true")
    run.add_argument("--expected-version", type=int)
    run.add_argument("--crash-after-completed", type=int)

    edits = subparsers.add_parser("run-edits", help="apply all edit batches with rebuild comparisons")
    edits.add_argument("--input", required=True, help="sample input directory")
    edits.add_argument("--workspace", required=True, help="experiment workspace")
    edits.add_argument("--reports", required=True, help="report output directory")
    edits.add_argument("--no-reset", action="store_true")
    edits.add_argument("--skip-full-comparison", action="store_true")

    demo = subparsers.add_parser("demo", help="generate inputs and run all verification checks")
    demo.add_argument("--root", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=20261120)
    demo.add_argument("--sample-dir", help="reuse an existing local sample directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        if Path(args.out).exists() and args.keep_existing:
            parser.error(f"{args.out} exists; remove --keep-existing to regenerate")
        summary = generate_sample(args.out, seed=args.seed, force=not args.keep_existing)
        _print_json(summary)
        return 0
    if args.command == "run":
        report_path = Path(args.report) if args.report else None
        try:
            result = run_pipeline(
                args.input,
                args.workspace,
                mode="full" if args.full else "incremental",
                fresh=bool(args.fresh or args.full),
                allow_config_migration=args.allow_config_migration,
                expected_global_version=args.expected_version,
                crash_after_completed=args.crash_after_completed,
                report_path=report_path,
                injections_path=args.injections,
            )
        except PipelineConfigMismatchError as exc:
            error_report = {
                "schema_version": 1,
                "business_code": exc.business_code,
                "message": exc.message,
                "resume_refused": True,
                **exc.details,
            }
            if report_path is not None:
                atomic_write_json(report_path, error_report)
            _print_json(error_report)
            return 2
        except SimulatedCrash as exc:
            crash_report = {
                "schema_version": 1,
                "business_code": exc.business_code,
                "message": exc.message,
                "exit_report_guaranteed": False,
                **exc.details,
            }
            _print_json(crash_report)
            return 3
        except PipelineError as exc:
            error_report = {
                "schema_version": 1,
                "business_code": exc.business_code,
                "message": exc.message,
                "resume_refused": False,
                **exc.details,
            }
            if report_path is not None:
                atomic_write_json(report_path, error_report)
            _print_json(error_report)
            return 2
        summary = {
            "snapshot": result.report.get("snapshot"),
            "validation_summary": result.report.get("validation_summary"),
            "failed_tasks": len(result.report.get("failures", [])),
            "reused": [item["task_name"] for item in result.report["lifecycle"]["reused"]],
            "retrained": [item["task_name"] for item in result.report["lifecycle"]["retrained"]],
        }
        _print_json(summary)
        return 0
    if args.command == "run-edits":
        verification = run_edit_sequence(
            args.input,
            args.workspace,
            args.reports,
            reset=not args.no_reset,
            compare_full=not args.skip_full_comparison,
        )
        _print_json(
            {
                "stage_comparisons": verification["stage_comparisons"],
                "idempotent_rerun": verification["idempotent_rerun"],
                "final_snapshot": verification["final_snapshot"],
            }
        )
        return 0
    if args.command == "demo":
        result = run_demo(args.root, seed=args.seed, sample_dir=args.sample_dir)
        _print_json(result)
        return 0 if result["all_checks_passed"] else 1
    parser.error("unknown command")
    return 2


__all__ = ["main"]

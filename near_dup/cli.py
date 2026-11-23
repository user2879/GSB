"""Command-line entry points for sample generation, validation, and demo."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .demo import run_demo
from .plan import load_edit_plan, load_injections
from .sample_data import generate_samples
from .text_store import load_text_jsonl


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="near_dup")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate deterministic samples")
    generate_parser.add_argument("--out", default="samples")

    demo_parser = subparsers.add_parser("demo", help="run the full verification demo")
    demo_parser.add_argument("--samples", default="samples")
    demo_parser.add_argument("--run-dir", default="runs/demo")
    demo_parser.add_argument("--report-dir", default="reports")

    validate_parser = subparsers.add_parser("validate", help="validate the sample inputs")
    validate_parser.add_argument("--samples", default="samples")

    args = parser.parse_args(argv)
    if args.command == "generate":
        _emit(generate_samples(args.out))
        return 0
    if args.command == "demo":
        report = run_demo(sample_dir=args.samples, run_dir=args.run_dir, report_dir=args.report_dir)
        _emit(
            {
                "report": f"{args.report_dir}/diagnostic_report.json",
                "incremental_vs_full_rebuild": report["incremental_vs_full_rebuild"],
                "concurrency_equal": report["concurrency"]["equal"],
                "recovery_equal": report["snapshot_recovery"]["resumed_equals_full"],
                "injection_hits": {
                    item["injection_id"]: item["hit"] for item in report["injections"]
                },
            }
        )
        return 0
    if args.command == "validate":
        config, failures = load_config(f"{args.samples}/config.yaml")
        result = {
            "config_hash": config.config_hash if config else None,
            "config_failures": [failure.to_dict() for failure in failures],
        }
        if config is not None:
            _, text_failures = load_text_jsonl(f"{args.samples}/texts.jsonl", config)
            _, plan_failures = load_edit_plan(f"{args.samples}/edit_plan.json", config)
            _, injection_failures = load_injections(f"{args.samples}/injections.json")
            result.update(
                {
                    "text_failures": [failure.to_dict() for failure in text_failures],
                    "plan_failures": [failure.to_dict() for failure in plan_failures],
                    "injection_failures": [failure.to_dict() for failure in injection_failures],
                }
            )
        _emit(result)
        return 0 if not failures else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


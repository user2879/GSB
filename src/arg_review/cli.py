from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .diagnostics import run_edit_equivalence, run_injection_suite
from .engine import ReviewEngine
from .samplegen import generate_sample_tree
from .state import load_workspace


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arg-review")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-samples")
    generate.add_argument("--root", default="samples")

    run = sub.add_parser("run")
    run.add_argument("--samples", default="samples")
    run.add_argument("--state-dir", default="artifacts")

    validate_samples = sub.add_parser("validate-samples")
    validate_samples.add_argument("--samples", default="samples")
    validate_samples.add_argument("--state-dir", default="artifacts/invalid-validation")

    verify = sub.add_parser("verify-edits")
    verify.add_argument("--samples", default="samples")
    verify.add_argument("--commands", default=None)
    verify.add_argument("--state-dir", default="artifacts/verification")
    verify.add_argument("--report", default="artifacts/edit_equivalence_report.json")

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--samples", default="samples")
    diagnose.add_argument("--state-dir", default="artifacts")

    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--samples", default="samples")
    all_cmd.add_argument("--state-dir", default="artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate-samples":
        paths = generate_sample_tree(args.root)
        for name, path in sorted(paths.items()):
            print(f"generated {name}: {path}")
        return 0

    if args.command == "run":
        state = load_workspace(args.samples)
        result = ReviewEngine(state, args.state_dir).run()
        report = result["report"]
        print(json.dumps({
            "status": report["status"],
            "snapshot_id": result["snapshot_commit"]["snapshot_id"],
            "projection_fingerprint": result["projection"]["projection_fingerprint"],
            "stage_completion": report["stage_completion"],
            "business_code_counts": report["business_code_counts"],
            "ledger_summary": report["ledger_summary"],
        }, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-samples":
        state = load_workspace(args.samples, include_invalid=True)
        result = ReviewEngine(state, args.state_dir).run()
        report = result["report"]
        invalid_count = report["business_code_counts"].get("E_SCHEMA_INVALID", 0)
        print(json.dumps({
            "status": report["status"],
            "e_schema_invalid": invalid_count,
            "completed_decisions": report["stage_completion"]["role_adjudication"][
                "completed_decisions"
            ],
            "business_code_counts": report["business_code_counts"],
        }, ensure_ascii=False, indent=2))
        return 0 if invalid_count >= 6 else 1

    if args.command == "verify-edits":
        commands = args.commands or str(Path(args.samples) / "edit_commands.jsonl")
        report = run_edit_equivalence(args.samples, commands, args.state_dir)
        write_json(Path(args.report), report)
        print(json.dumps({
            "all_steps_equal": report["all_steps_equal"],
            "completed_steps": report["completed_steps"],
            "report": args.report,
        }, ensure_ascii=False, indent=2))
        return 0 if report["all_steps_equal"] else 1

    if args.command == "diagnose":
        state = load_workspace(args.samples)
        main_result = ReviewEngine(state, args.state_dir).run()
        suite = run_injection_suite(state, Path(args.state_dir) / "injection-runs")
        aggregate = {
            "baseline_report": main_result["report"],
            "injection_suite": suite,
        }
        report_path = Path(args.state_dir) / "diagnostic_report.json"
        write_json(report_path, aggregate)
        print(json.dumps({
            "diagnostic_report": str(report_path),
            "all_injections_hit": suite["all_injections_hit"],
            "hit_count": suite["hit_count"],
            "injection_count": suite["injection_count"],
            "baseline_status": main_result["report"]["status"],
        }, ensure_ascii=False, indent=2))
        return 0 if suite["all_injections_hit"] else 1

    if args.command == "all":
        generate_sample_tree(args.samples)
        state = load_workspace(args.samples)
        main_result = ReviewEngine(state, args.state_dir).run()
        equivalence = run_edit_equivalence(
            args.samples,
            Path(args.samples) / "edit_commands.jsonl",
            Path(args.state_dir) / "verification",
        )
        suite = run_injection_suite(state, Path(args.state_dir) / "injection-runs")
        write_json(Path(args.state_dir) / "edit_equivalence_report.json", equivalence)
        aggregate = {
            "baseline_report": main_result["report"],
            "edit_equivalence": {
                "all_steps_equal": equivalence["all_steps_equal"],
                "completed_steps": equivalence["completed_steps"],
            },
            "injection_suite": suite,
        }
        write_json(Path(args.state_dir) / "diagnostic_report.json", aggregate)
        print(json.dumps({
            "run": "ok",
            "equivalence": equivalence["all_steps_equal"],
            "injections": suite["all_injections_hit"],
            "projection_fingerprint": main_result["projection"]["projection_fingerprint"],
        }, ensure_ascii=False, indent=2))
        return 0 if equivalence["all_steps_equal"] and suite["all_injections_hit"] else 1

    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

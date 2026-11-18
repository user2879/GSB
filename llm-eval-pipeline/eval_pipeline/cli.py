"""Command line entry points for generation, execution and verification."""

from __future__ import annotations

import argparse
import json
import os
import shutil

from .config import load_config, load_rules
from .generator import generate_all
from .loader import load_evaluations, load_preferences
from .pipeline import Pipeline, normalized_report
from .util import canonical_json, read_json


def _load_inputs(sample_dir: str, config_name: str = "config.yaml",
                 rules_name: str = "rules.json"):
    failures = []
    config = load_config(os.path.join(sample_dir, config_name))
    rules = load_rules(os.path.join(sample_dir, rules_name), failures)
    preferences = load_preferences(os.path.join(sample_dir, "preferences.jsonl"), failures)
    evaluations = load_evaluations(os.path.join(sample_dir, "evaluations.jsonl"), failures)
    return config, rules, preferences, evaluations, failures


def cmd_generate(args) -> None:
    paths = generate_all(args.sample_dir, args.preferences, args.evaluations, args.seed)
    print(json.dumps(paths, indent=2, ensure_ascii=False))


def cmd_run(args) -> None:
    config, rules, preferences, evaluations, load_failures = _load_inputs(
        args.sample_dir, args.config, args.rules
    )
    pipeline = Pipeline(
        config,
        rules,
        preferences,
        evaluations,
        args.workdir,
        injection_path=args.injection,
        strict_resume=args.strict_resume,
        crash_after=args.crash_after,
    )
    pipeline.failures.extend(load_failures)
    report = pipeline.run()
    print(f"report written to {os.path.join(args.workdir, 'report.json')}")
    print(json.dumps({
        "tier": report["tier"],
        "strategy": report["strategy"]["strategy_id"],
        "hallucination_rate": report["hallucination"]["detection_rate"],
        "failures": len(report["failures"]),
        "cache": {
            "reuse": report["cache"]["reuse_count"],
            "recompute": report["cache"]["recompute_count"],
            "invalidated": report["cache"]["invalidated_count"],
        },
    }, indent=2, ensure_ascii=False))


def cmd_verify(args) -> None:
    left = read_json(args.report_a)
    right = read_json(args.report_b)
    if args.normalized:
        left = normalized_report(left)
        right = normalized_report(right)
    if canonical_json(left) == canonical_json(right):
        mode = "normalized (core result fields)" if args.normalized else "full report"
        print(f"FIELD-IDENTICAL: reports match after canonicalization ({mode})")
        return
    raise SystemExit("MISMATCH: reports differ")


def cmd_clean(args) -> None:
    if os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)
    print(f"removed {args.workdir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic LLM eval pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic local samples")
    gen.add_argument("--sample-dir", required=True)
    gen.add_argument("--preferences", type=int, default=600)
    gen.add_argument("--evaluations", type=int, default=300)
    gen.add_argument("--seed", type=int, default=42)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="run the pipeline")
    run.add_argument("--sample-dir", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--rules", default="rules.json")
    run.add_argument("--injection", default=None)
    run.add_argument("--strict-resume", action="store_true")
    run.add_argument("--crash-after", type=int, default=None)
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="compare two reports for field equality")
    verify.add_argument("report_a")
    verify.add_argument("report_b")
    verify.add_argument("--normalized", action="store_true",
                        help="ignore run-state audit counters such as cache reuse counts")
    verify.set_defaults(func=cmd_verify)

    clean = sub.add_parser("clean", help="remove a work directory")
    clean.add_argument("--workdir", required=True)
    clean.set_defaults(func=cmd_clean)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)

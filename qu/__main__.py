"""Command line entry point: data generation and experiment execution."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_CONFIG, PipelineConfig
from .experiment import ExperimentRunner, TierSpec
from .loader import load_bundle
from .synthetic import DEFAULT_TIERS, SyntheticGenerator


def _print(payload: dict) -> None:
    # Explicit UTF-8 keeps Windows consoles and redirected files consistent.
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def cmd_generate(args) -> int:
    generator = SyntheticGenerator(args.seed)
    tiers = DEFAULT_TIERS
    if args.tiers:
        wanted = set(args.tiers.split(","))
        tiers = tuple(spec for spec in DEFAULT_TIERS if spec.tier_id in wanted)
    manifest = {}
    for spec in tiers:
        manifest[spec.tier_id] = generator.generate_tier(args.data_dir, spec)
    with open(f"{args.data_dir}/manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _print({"generated_tiers": [spec.tier_id for spec in tiers],
            "data_dir": args.data_dir})
    return 0


def cmd_run(args) -> int:
    config = PipelineConfig(
        seed=args.seed,
        max_workers=args.workers,
        latency_budget_ms=args.budget_ms,
        max_rewrites=args.max_rewrites,
        exhaustive_verify_sample=args.verify_sample,
    )
    with open(f"{args.data_dir}/manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    canonical = [spec.tier_id for spec in DEFAULT_TIERS]
    tier_ids = (args.tiers.split(",") if args.tiers
                else [tid for tid in canonical if tid in manifest])
    specs = []
    for tier_id in tier_ids:
        entry = manifest[tier_id]
        specs.append(TierSpec(
            tier_id=tier_id,
            samples_path=entry["samples"],
            vocab_path=entry["vocab"],
            entity_path=entry["entities"],
            rules_path=entry["rules"],
        ))

    runner = ExperimentRunner(config, args.work_dir)
    runner.install_signal_handlers()

    def loader(spec: TierSpec):
        return load_bundle(spec.samples_path, spec.vocab_path,
                           spec.entity_path, spec.rules_path)

    report = runner.run_experiment(specs, loader)
    _print({
        "report_path": f"{args.work_dir}/experiment_report.json",
        "tiers": [tier["tier_id"] for tier in report["tiers"]],
        "all_tiers_match_exhaustive": report["all_tiers_match_exhaustive"],
        "all_concurrency_comparisons_equal":
            report["all_concurrency_comparisons_equal"],
        "graceful_stop": report["interrupted_graceful_stop"],
        "exit_code": 0,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qu", description="Local e-commerce query understanding service")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic synthetic data")
    gen.add_argument("--data-dir", default="data/generated")
    gen.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    gen.add_argument("--tiers", default="")
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="run multi-scale experiments")
    run.add_argument("--data-dir", default="data/generated")
    run.add_argument("--work-dir", default="artifacts/run")
    run.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    run.add_argument("--workers", type=int, default=DEFAULT_CONFIG.max_workers)
    run.add_argument("--budget-ms", type=float,
                     default=DEFAULT_CONFIG.latency_budget_ms)
    run.add_argument("--max-rewrites", type=int, default=DEFAULT_CONFIG.max_rewrites)
    run.add_argument("--verify-sample", type=int,
                     default=DEFAULT_CONFIG.exhaustive_verify_sample)
    run.add_argument("--tiers", default="")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry points for generation, validation and builds."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .canonical import write_json
from .config import load_project, project_config_hash
from .engine import run_build
from .sample import generate_sample


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def command_generate(args: argparse.Namespace) -> int:
    path = generate_sample(args.out, seed=args.seed, port_start=args.port_start)
    _print({"generated": str(path), "seed": args.seed})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    project, issues = load_project(args.config)
    output = {
        "config": str(args.config),
        "config_hash": project_config_hash(project),
        "documents": len(project.documents),
        "shards": [shard.to_dict() for shard in project.shards],
        "current_revision": project.current_revision,
        "compression": project.profile_for_revision(project.current_revision).to_dict(),
        "issues": issues,
        "valid": not issues,
    }
    _print(output)
    return 0 if not issues else 2


def command_build(args: argparse.Namespace) -> int:
    if args.fresh and Path(args.workdir).exists():
        shutil.rmtree(args.workdir)
    result = run_build(
        args.config,
        args.workdir,
        mode=args.mode,
        workers=args.workers,
        global_version=args.global_version,
        prior_pending=args.prior_pending,
        start_services=not args.no_services,
        commit=not (args.stage_only or args.simulate_termination),
    )
    destination = Path(args.result_file) if args.result_file else Path(args.workdir) / "build-result.json"
    write_json(destination, result.to_dict())
    summary = {
        "mode": result.mode,
        "global_version": result.global_version,
        "snapshot_id": result.snapshot_id,
        "committed": result.committed,
        "staging_path": result.staging_path,
        "entries_count": result.ledger["entries_count"],
        "shard_entry_counts": result.ledger["shard_entry_counts"],
        "arbitration": result.ledger["arbitration_stats"],
        "business_code_counts": result.report["business_code_counts"],
        "failure_count": len(result.failure_list),
        "pending_shards": result.report["pending_shards"],
        "recovery": result.recovery,
        "reuse_counts": result.report["reuse"]["counts"],
        "end_to_end_ms_monotonic": result.report["end_to_end_ms_monotonic"],
        "result_file": str(destination),
    }
    _print(summary)
    if args.simulate_termination:
        return 9
    return 0


def command_demo(args: argparse.Namespace) -> int:
    sample_dir = Path(args.out) / "sample"
    if args.fresh and sample_dir.exists():
        shutil.rmtree(sample_dir)
    config = generate_sample(sample_dir, seed=args.seed, port_start=args.port_start)
    baseline = Path(args.out) / "baseline"
    if args.fresh and baseline.exists():
        shutil.rmtree(baseline)
    result = run_build(config, baseline, mode="full", workers=3, global_version=1)
    _print(
        {
            "sample": str(sample_dir),
            "baseline_workdir": str(baseline),
            "snapshot_id": result.snapshot_id,
            "entries_count": result.ledger["entries_count"],
            "shard_entry_counts": result.ledger["shard_entry_counts"],
            "arbitration": result.ledger["arbitration_stats"],
            "failure_list": result.failure_list,
            "recall_selected": [
                {
                    "query_id": item["query_id"],
                    "selected": [candidate["doc_id"] for candidate in item["selected"]],
                }
                for item in result.recall
            ],
        }
    )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    from .verify import run_verification

    report = run_verification(args.out, seed=args.seed)
    _print(report)
    return 0 if report["all_checks_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsb-index", description="Deterministic sharded inverted-index prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local sample files")
    generate.add_argument("--out", default="samples/seed-17")
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument("--port-start", type=int, default=19200)
    generate.set_defaults(func=command_generate)

    validate = subparsers.add_parser("validate", help="validate a project bundle without starting services")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=command_validate)

    build = subparsers.add_parser("build", help="run a full or incremental build")
    build.add_argument("--config", required=True)
    build.add_argument("--workdir", required=True)
    build.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--global-version", type=int)
    build.add_argument("--prior-pending", nargs="*", default=None)
    build.add_argument("--no-services", action="store_true", help="validation/dispatch diagnostic without spawned workers")
    build.add_argument("--stage-only", action="store_true", help="write a staging snapshot but do not publish it")
    build.add_argument("--simulate-termination", action="store_true", help="stage and exit with rc=9 before commit")
    build.add_argument("--fresh", action="store_true")
    build.add_argument("--result-file")
    build.set_defaults(func=command_build)

    demo = subparsers.add_parser("demo", help="generate a sample and run a real loopback baseline")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--port-start", type=int, default=19200)
    demo.add_argument("--fresh", action="store_true")
    demo.set_defaults(func=command_demo)

    verify = subparsers.add_parser("verify", help="run the complete acceptance verification suite")
    verify.add_argument("--out", default="artifacts/verification")
    verify.add_argument("--seed", type=int, default=17)
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry points: generate, run, rebuild, inject."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import WorkbenchError
from .generator import build_default_dataset
from .injection import run_injection_suite
from .pipeline import run_pipeline
from .rebuild import compare_outcomes, full_rebuild
from .utils import read_json, write_json_atomic


def _print_json(value: object) -> None:
    json.dump(value, sys.stdout, sort_keys=True, indent=2, default=str)
    sys.stdout.write("\n")


def cmd_generate(args: argparse.Namespace) -> int:
    result = build_default_dataset(args.input_dir, seed=args.seed)
    _print_json(
        {
            "samples_path": result["samples_path"],
            "tasks_path": result["tasks_path"],
            "base_versions": [
                result["base_v1"]["base_version"],
                result["base_v2"]["base_version"],
            ],
            "base_fingerprints": {
                result["base_v1"]["base_version"]: result["base_v1"][
                    "base_fingerprint"
                ],
                result["base_v2"]["base_version"]: result["base_v2"][
                    "base_fingerprint"
                ],
            },
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    outcome = run_pipeline(
        input_dir=args.input_dir,
        workspace=args.workspace,
        config=config,
        base_subdir=args.base_subdir,
        strict_resume=args.strict,
    )
    summary = {
        "version": outcome["manifest"]["version"],
        "trained_tasks": outcome["trained_tasks"],
        "reused_tasks": outcome["reused_tasks"],
        "invalidation": outcome["invalidation"],
        "failures": outcome["failures"],
        "validation": outcome["validation_summary"]["entries"],
    }
    _print_json(summary)
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    config = read_json(args.config)
    # Reuse the previous manifest identities so dependency verdicts are identical.
    baseline = None
    manifest_path = f"{args.workspace}/current/manifest.json"
    try:
        baseline = read_json(manifest_path).get("task_identities")
    except OSError:
        baseline = None
    incremental = run_pipeline(
        input_dir=args.input_dir,
        workspace=args.workspace,
        config=config,
        base_subdir=args.base_subdir,
    )
    rebuild = full_rebuild(
        input_dir=args.input_dir,
        fresh_workspace=args.rebuild_workspace,
        config=config,
        baseline_identities=baseline,
        base_subdir=args.base_subdir,
    )
    comparison = compare_outcomes(incremental, rebuild)
    _print_json(
        {
            "incremental_version": incremental["manifest"]["version"],
            "comparison": comparison,
            "incremental_invalidation": incremental["invalidation"],
            "rebuild_invalidation": rebuild["invalidation"],
        }
    )
    return 0 if comparison["equal"] else 2


def cmd_inject(args: argparse.Namespace) -> int:
    report = run_injection_suite(args.report_root)
    _print_json(report)
    return 0 if report["hit_count"] == report["total"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-bench",
        description="Deterministic local LoRA incremental training workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic dataset")
    gen.add_argument("input_dir")
    gen.add_argument("--seed", type=int, default=20240501)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="run incremental pipeline")
    run.add_argument("input_dir")
    run.add_argument("workspace")
    run.add_argument("--config", required=True)
    run.add_argument("--base-subdir", default="base_v1")
    run.add_argument("--strict", action="store_true")
    run.set_defaults(func=cmd_run)

    rebuild = sub.add_parser("rebuild", help="incremental vs full rebuild check")
    rebuild.add_argument("input_dir")
    rebuild.add_argument("workspace")
    rebuild.add_argument("rebuild_workspace")
    rebuild.add_argument("--config", required=True)
    rebuild.add_argument("--base-subdir", default="base_v1")
    rebuild.set_defaults(func=cmd_rebuild)

    inject = sub.add_parser("inject", help="run fault injection suite")
    inject.add_argument("report_root")
    inject.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WorkbenchError as exc:
        payload = {"code": exc.code, "stage": exc.stage, "message": exc.message}
        write_json_atomic(".lora_bench_last_error.json", payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

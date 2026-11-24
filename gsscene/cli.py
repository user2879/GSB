"""Command line entry points for sample generation, runs and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_cache_arrays, compare_reports
from .experiment import main as experiment_main, run_equivalence_experiment
from .pipeline import ScenePipeline, compact_report
from .sample import generate_sample
from .util import atomic_write_json
from .validation import load_project, materialize_injected_sample


def _cmd_generate(args: argparse.Namespace) -> int:
    out = generate_sample(
        args.out,
        width=args.width,
        height=args.height,
        views=args.views,
        seed=args.seed,
        concurrency=args.concurrency,
        force=args.force,
    )
    print(out)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    project = load_project(args.sample)
    payload = {
        "active_views": sorted(project.views),
        "config_fingerprint": project.config.fingerprint() if project.config else None,
        "failures": [failure.as_dict() for failure in project.failures],
        "injection_hits": project.injection_hits,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not any(failure.severity == "ERROR" for failure in project.failures) else 1


def _cmd_run(args: argparse.Namespace) -> int:
    pipeline = ScenePipeline(
        args.sample,
        args.work,
        replay_edits=not args.no_replay_edits,
        force_rebuild=args.force,
        worker_override=args.workers,
    )
    report = pipeline.run(max_new_edits=args.max_new_edits)
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(compact_report(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left_report).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right_report).read_text(encoding="utf-8"))
    result = {"reports": compare_reports(left, right)}
    if args.left_work and args.right_work:
        result["arrays"] = compare_cache_arrays(args.left_work, args.right_work)
    result["ok"] = all(value.get("equal", False) for value in result.values())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _cmd_inject(args: argparse.Namespace) -> int:
    materialize_injected_sample(args.sample, args.out, selected_injection_ids=set(args.ids) or None)
    print(args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsscene", description="Deterministic multi-resolution Gaussian scene pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic local sample")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--width", type=int, default=64)
    generate.add_argument("--height", type=int, default=64)
    generate.add_argument("--views", type=int, default=6)
    generate.add_argument("--seed", type=int, default=20240517)
    generate.add_argument("--concurrency", type=int, default=4)
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=_cmd_generate)

    validate = subparsers.add_parser("validate", help="validate a sample directory")
    validate.add_argument("--sample", type=Path, required=True)
    validate.set_defaults(func=_cmd_validate)

    run = subparsers.add_parser("run", help="execute or resume a pipeline workspace")
    run.add_argument("--sample", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--report", type=Path)
    run.add_argument("--max-new-edits", type=int, default=None)
    run.add_argument("--workers", type=int, default=None, help="execution cap; never exceeds config concurrency")
    run.add_argument("--force", action="store_true", help="discard state/cache/snapshots and rebuild")
    run.add_argument("--no-replay-edits", action="store_true", help="treat the sample as already finalized")
    run.set_defaults(func=_cmd_run)

    compare = subparsers.add_parser("compare", help="compare two report JSON files and optionally cache arrays")
    compare.add_argument("--left-report", type=Path, required=True)
    compare.add_argument("--right-report", type=Path, required=True)
    compare.add_argument("--left-work", type=Path)
    compare.add_argument("--right-work", type=Path)
    compare.set_defaults(func=_cmd_compare)

    inject = subparsers.add_parser("inject", help="materialize an isolated injected sample copy")
    inject.add_argument("--sample", type=Path, required=True)
    inject.add_argument("--out", type=Path, required=True)
    inject.add_argument("--ids", nargs="*", default=[])
    inject.set_defaults(func=_cmd_inject)

    experiment = subparsers.add_parser("experiment", help="run the full incremental/full/serial acceptance experiment")
    experiment.add_argument("--sample", type=Path, default=Path("artifacts/run/sample"))
    experiment.add_argument("--out", type=Path, default=Path("artifacts/run/experiment"))
    experiment.add_argument("--regenerate", action="store_true")
    experiment.add_argument("--width", type=int, default=64)
    experiment.add_argument("--height", type=int, default=64)
    experiment.add_argument("--views", type=int, default=6)
    experiment.set_defaults(func=lambda args: experiment_main(["--sample", str(args.sample), "--out", str(args.out), "--width", str(args.width), "--height", str(args.height), "--views", str(args.views)] + (["--regenerate"] if args.regenerate else [])))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

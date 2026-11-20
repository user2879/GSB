"""Command-line entry points for the local deterministic pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import SCHEMA_VERSION
from .generate import generate_dataset, replace_sample
from .pipeline import RunOptions, run_pipeline
from .store import SnapshotStore
from .util import atomic_write_json, read_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synthml",
        description="Deterministic synthetic multi-label image classification pipeline",
    )
    parser.add_argument("--version", action="version", version="synthml 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic samples and configurations")
    generate.add_argument("--out", type=Path, default=Path("data/samples"))
    generate.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    generate.add_argument("--plan", type=Path, default=Path("config/scale_plan.json"))
    generate.add_argument("--injections", type=Path, default=Path("config/injections.yaml"))
    generate.add_argument(
        "--serial-config", type=Path, default=Path("config/experiment_serial.yaml")
    )
    generate.add_argument("--samples", type=int, default=72)
    generate.add_argument("--seed", type=int, default=20260317)
    generate.add_argument("--width", type=int, default=96)
    generate.add_argument("--height", type=int, default=96)
    generate.add_argument("--noise", type=float, default=4.0)
    generate.add_argument("--blur", type=float, default=0.55)
    generate.add_argument("--concurrency", type=int, default=4)
    generate.add_argument("--snapshot-interval", type=int, default=8)

    run = subparsers.add_parser("run", help="run incremental, rebuild or resume pipeline")
    run.add_argument("--samples", type=Path, default=Path("data/samples"))
    run.add_argument("--config", type=Path, default=Path("config/experiment.yaml"))
    run.add_argument("--plan", type=Path, default=Path("config/scale_plan.json"))
    run.add_argument("--state", type=Path, default=Path("state"))
    run.add_argument("--report", type=Path, default=Path("reports/report.json"))
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--incremental", action="store_const", const="incremental", dest="mode")
    mode.add_argument("--rebuild", action="store_const", const="rebuild", dest="mode")
    mode.add_argument("--resume", action="store_const", const="resume", dest="mode")
    run.set_defaults(mode="incremental")
    run.add_argument("--tiers", nargs="+", help="subset of scale tier ids; default is all tiers")
    run.add_argument("--injections", type=Path, default=None)
    run.add_argument("--injection-id", action="append", dest="injection_ids")
    run.add_argument("--strategy", dest="force_strategy")
    run.add_argument("--feature-method", dest="force_feature_method")
    run.add_argument("--color-bins", type=int, dest="force_color_bins")
    run.add_argument("--hog-bins", type=int, dest="force_hog_bins")
    run.add_argument("--classifier", dest="force_classifier")
    run.add_argument("--knn-k", type=int, dest="force_knn_k")
    run.add_argument("--threshold", type=float, dest="force_threshold")
    run.add_argument("--precision")
    run.add_argument("--concurrency", type=int)
    run.add_argument("--snapshot-interval", type=int)

    replace = subparsers.add_parser("replace-sample", help="replace one sample with another sample pair")
    replace.add_argument("--samples", type=Path, default=Path("data/samples"))
    replace.add_argument("--target", required=True, help="target PNG filename inside --samples")
    replace.add_argument("--source", type=Path, required=True, help="source PNG with matching JSON")

    report = subparsers.add_parser("report", help="export the report from the current snapshot")
    report.add_argument("--state", type=Path, default=Path("state"))
    report.add_argument("--out", type=Path, default=Path("reports/report.json"))
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "generate":
        result = generate_dataset(
            args.out,
            args.config,
            args.plan,
            sample_count=args.samples,
            seed=args.seed,
            width=args.width,
            height=args.height,
            noise=args.noise,
            blur=args.blur,
            concurrency=args.concurrency,
            snapshot_interval=args.snapshot_interval,
            injection_path=args.injections,
            serial_config_path=args.serial_config,
        )
        _print_json(result)
        return 0
    if args.command == "replace-sample":
        result = replace_sample(args.samples, args.target, args.source)
        _print_json(result)
        return 0
    if args.command == "report":
        store = SnapshotStore(args.state, SCHEMA_VERSION)
        current = store.load_current()
        if current is None:
            print("E_CONFIG_MISMATCH: no current snapshot", file=sys.stderr)
            return 3
        report_path = current.path / "report.json"
        if not report_path.exists():
            print("E_CONFIG_MISMATCH: current snapshot has no completed report", file=sys.stderr)
            return 3
        report = read_json(report_path)
        atomic_write_json(args.out, report)
        _print_json({"report": str(args.out), "snapshot": current.name})
        return 0
    if args.command == "run":
        options = RunOptions(
            samples_dir=args.samples,
            config_path=args.config,
            plan_path=args.plan,
            state_dir=args.state,
            report_path=args.report,
            mode=args.mode,
            tiers=args.tiers,
            injections_path=args.injections,
            injection_ids=args.injection_ids,
            force_strategy=args.force_strategy,
            force_feature_method=args.force_feature_method,
            force_color_bins=args.force_color_bins,
            force_hog_bins=args.force_hog_bins,
            force_classifier=args.force_classifier,
            force_knn_k=args.force_knn_k,
            force_threshold=args.force_threshold,
            precision=args.precision,
            concurrency=args.concurrency,
            snapshot_interval=args.snapshot_interval,
        )
        result = run_pipeline(options)
        summary = {
            "exit_code": result.exit_code,
            "report": str(result.report_path),
            "snapshot": result.report.get("snapshot"),
            "failures": len(result.report.get("failures", [])),
            "tiers": result.report.get("scope", {}).get("tier_ids", []),
            "diagnosis": result.report.get("diagnosis", []),
        }
        _print_json(summary)
        return result.exit_code
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

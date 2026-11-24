from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_and_validate_config
from .engine import run_pipeline
from .errors import sorted_failures
from .faults import FaultInjector
from .generator import write_example_bundle
from .generator import write_injection_bundle
from .samples import load_and_validate_samples
from .snapshot import atomic_write_json, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibpipe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic example config and JSONL")
    generate.add_argument("--out", default="examples/local_sample")
    generate.add_argument("--injections", default="examples/injections")
    generate.add_argument("--no-injections", action="store_true")

    validate = sub.add_parser("validate", help="validate config and samples")
    validate.add_argument("--config", required=True)
    validate.add_argument("--samples", required=True)
    validate.add_argument("--injection-id")
    validate.add_argument("--report")

    run = sub.add_parser("run", help="run the deterministic pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--samples", required=True)
    run.add_argument("--out", default="out")
    run.add_argument("--workers", type=int)
    run.add_argument("--budget", type=int)
    run.add_argument("--gear")
    run.add_argument("--inject-runtime")
    run.add_argument("--no-attribution", action="store_true")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--report-name", default="pipeline_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config_path, samples_path = write_example_bundle(args.out)
        injection_files = []
        if not args.no_injections:
            injection_files = [str(p) for p in write_injection_bundle(args.out, args.injections)]
        print(json.dumps({"config": str(config_path), "samples": str(samples_path), "injections": injection_files}, ensure_ascii=False))
        return 0
    if args.command == "validate":
        return _validate(args)
    if args.command == "run":
        return _run(args)
    return 2


def _validate(args: argparse.Namespace) -> int:
    config, config_failures = load_and_validate_config(args.config, args.injection_id)
    failures = list(config_failures)
    sample_count = 0
    sample_fingerprint = ""
    if config is not None:
        samples, sample_failures, sample_fingerprint = load_and_validate_samples(
            args.samples,
            len(config.channels),
            injection_id=args.injection_id,
            channel_names=config.channels,
        )
        failures.extend(sample_failures)
        sample_count = len(samples)
    report = {
        "status": "valid" if not failures else "invalid",
        "injection_id": args.injection_id,
        "config": args.config,
        "samples": args.samples,
        "sample_count": sample_count,
        "sample_fingerprint": sample_fingerprint,
        "failures": sorted_failures(failures),
    }
    if args.report:
        atomic_write_json(args.report, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def _run(args: argparse.Namespace) -> int:
    config, config_failures = load_and_validate_config(args.config)
    if config is None:
        report = {"status": "invalid", "failures": sorted_failures(config_failures)}
        write_report(args.out, args.report_name, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    samples, sample_failures, sample_fp = load_and_validate_samples(
        args.samples, len(config.channels), channel_names=config.channels
    )
    if sample_failures:
        report = {
            "status": "invalid",
            "config_fingerprint": config.fingerprint,
            "sample_fingerprint": sample_fp,
            "failures": sorted_failures(sample_failures),
        }
        write_report(args.out, args.report_name, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    injector = FaultInjector()
    if args.inject_runtime:
        import json as _json

        injector = FaultInjector.from_dict(_json.loads(Path(args.inject_runtime).read_text(encoding="utf-8")))

    result = run_pipeline(
        config,
        samples,
        args.out,
        max_workers=args.workers,
        budget=args.budget,
        gear_override=args.gear,
        include_attribution=not args.no_attribution,
        use_cache=not args.no_cache,
        fault_injector=injector,
    )
    report = result["report"]
    report_path = write_report(args.out, args.report_name, report)
    atomic_write_json(Path(args.out) / "stage_products.json", result["products"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "summary": report["summary"],
                "failures": len(report["failures"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] in {"ok", "completed_with_failures", "budget_pending"} else 1


if __name__ == "__main__":
    sys.exit(main())

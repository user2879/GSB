"""Command-line interface for sample generation and pipeline execution."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .pipeline import PIPELINE_VERSION, QuantizationPipeline
from .samples import DEFAULT_SEED, generate_samples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-quant-pipeline",
        description="Deterministic LLM weight quantization, distillation correction, and diagnostics",
    )
    parser.add_argument("--version", action="version", version=PIPELINE_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="generate deterministic local fixtures")
    generate.add_argument("--sample-dir", default="samples")
    generate.add_argument("--config-dir", default="configs")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--sample-count", type=int, default=4)

    run = subparsers.add_parser("run", help="run the complete quantization pipeline")
    run.add_argument("--sample-dir", default="samples")
    run.add_argument("--output-dir", default="artifacts/run")
    run.add_argument("--weights")
    run.add_argument("--calibration")
    run.add_argument("--config")
    run.add_argument("--injections")
    run.add_argument("--profile", help="force a declared quantization profile for every layer")
    run.add_argument("--fractional-bits", type=int, help="force fixed-point fractional bit width")
    run.add_argument("--max-workers", type=int)
    run.add_argument("--point-budget", type=int)
    run.add_argument("--layer-point-quota", type=int)
    run.add_argument("--snapshot-interval", type=int)
    run.add_argument("--quiet", action="store_true")

    verify = subparsers.add_parser(
        "verify", help="run once and then verify a full-from-scratch build matches the result fingerprint"
    )
    verify.add_argument("--sample-dir", default="samples")
    verify.add_argument("--output-dir", default="artifacts/verify_incremental")
    verify.add_argument("--fresh-dir", default="artifacts/verify_fresh")
    verify.add_argument("--injections")
    verify.add_argument("--profile")
    verify.add_argument("--fractional-bits", type=int)
    verify.add_argument("--max-workers", type=int)
    verify.add_argument("--point-budget", type=int)
    verify.add_argument("--layer-point-quota", type=int)
    verify.add_argument("--snapshot-interval", type=int)
    return parser


def _install_signal_handlers(stop_event: threading.Event) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass


def _run_command(args: argparse.Namespace) -> int:
    stop_event = threading.Event()
    previous = _install_signal_handlers(stop_event)
    try:
        pipeline = QuantizationPipeline(
            args.sample_dir,
            args.output_dir,
            weights_path=args.weights,
            calibration_path=args.calibration,
            config_path=args.config,
            injections_path=args.injections,
            profile=args.profile,
            fractional_bits=args.fractional_bits,
            max_workers=args.max_workers,
            point_budget=args.point_budget,
            layer_point_quota=args.layer_point_quota,
            snapshot_interval=args.snapshot_interval,
            stop_event=stop_event,
        )
        report = pipeline.run()
    finally:
        _restore_signal_handlers(previous)
    if not args.quiet:
        _print_report_summary(report, Path(args.output_dir) / "report.json")
    if report["status"] == "completed" and not report["pending"] and not report["failures"]:
        return 0
    if report["status"] == "partial" and report.get("pending"):
        return 3
    return 2


def _verify_command(args: argparse.Namespace) -> int:
    common = {
        "injections_path": args.injections,
        "profile": args.profile,
        "fractional_bits": args.fractional_bits,
        "max_workers": args.max_workers,
        "point_budget": args.point_budget,
        "layer_point_quota": args.layer_point_quota,
        "snapshot_interval": args.snapshot_interval,
    }
    incremental = QuantizationPipeline(args.sample_dir, args.output_dir, **common).run()
    second = QuantizationPipeline(args.sample_dir, args.output_dir, **common).run()
    fresh = QuantizationPipeline(args.sample_dir, args.fresh_dir, **common).run()
    compared = {
        "first": incremental["result_fingerprint"],
        "second_incremental": second["result_fingerprint"],
        "fresh_full": fresh["result_fingerprint"],
        "incremental_reports_first": {
            "reused": incremental["cache_ledger"]["reused"],
            "recomputed": incremental["cache_ledger"]["recomputed"],
        },
        "incremental_reports_second": {
            "reused": second["cache_ledger"]["reused"],
            "recomputed": second["cache_ledger"]["recomputed"],
        },
        "fresh_reports": {
            "reused": fresh["cache_ledger"]["reused"],
            "recomputed": fresh["cache_ledger"]["recomputed"],
        },
        "match": incremental["result_fingerprint"]
        == second["result_fingerprint"]
        == fresh["result_fingerprint"],
    }
    print(json_dumps(compared))
    return 0 if compared["match"] else 1


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _print_report_summary(report: dict[str, Any], path: Path) -> None:
    print(
        json_dumps(
            {
                "status": report["status"],
                "report": str(path),
                "result_fingerprint": report.get("result_fingerprint"),
                "business_codes": report.get("business_codes", []),
                "layer_status": {
                    layer["name"]: layer["status"] for layer in report.get("layers", [])
                },
                "pending": report.get("pending", []),
                "failures": report.get("failures", []),
                "cache_ledger": {
                    "reused": report.get("cache_ledger", {}).get("reused", []),
                    "recomputed": report.get("cache_ledger", {}).get("recomputed", []),
                },
                "injections": report.get("injections", []),
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        paths = generate_samples(
            args.sample_dir,
            args.config_dir,
            seed=args.seed,
            sample_count=args.sample_count,
        )
        print(json_dumps(paths))
        return 0
    if args.command == "run":
        return _run_command(args)
    if args.command == "verify":
        return _verify_command(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


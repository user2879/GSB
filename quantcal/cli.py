"""Command line entry point for the local deterministic demo."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import ERROR_EXIT_CODES, QuantCalError
from .pipeline import run_pipeline
from .util import atomic_write_json, load_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantisation range calibration and saturation pipeline")
    parser.add_argument("--input-dir", required=True, help="directory containing tensors/config/exchanges")
    parser.add_argument("--work-dir", required=True, help="snapshot, cache and output directory")
    parser.add_argument("--exchange-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--report", default=None, help="optional diagnostics report path")
    parser.add_argument("--resume", action="store_true", help="strictly resume the committed snapshot")
    parser.add_argument("--force-rebuild", action="store_true", help="ignore caches and rebuild from empty state")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--batch-point-budget", type=int, default=None)
    parser.add_argument(
        "--injection-report",
        default=None,
        help="JSON file containing injection hit objects to copy into diagnostics",
    )
    return parser


def _print_json(value: Any) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    injection_hits = None
    try:
        if args.injection_report:
            document = load_json(args.injection_report)
            if isinstance(document, dict) and isinstance(document.get("hits"), list):
                injection_hits = document["hits"]
            elif isinstance(document, list):
                injection_hits = document
            elif isinstance(document, dict):
                injection_hits = [document]
            else:
                raise QuantCalError(
                    "E_FORMAT_INVALID", "injection report must be an object or array", stage="parse"
                )
        result = run_pipeline(
            input_dir=args.input_dir,
            work_dir=args.work_dir,
            exchange_version=args.exchange_version,
            report_path=args.report,
            resume=args.resume,
            force_rebuild=args.force_rebuild,
            max_concurrency=args.max_concurrency,
            batch_point_budget=args.batch_point_budget,
            injection_hits=injection_hits,
        )
        _print_json(
            {
                "exit_code": result.exit_code,
                "summary": result.result["summary"],
                "errors": result.errors,
                "pending": result.pending,
                "output_dir": str(result.output_dir),
                "snapshot_version": (
                    result.snapshot.get("global_version") if result.snapshot else None
                ),
            }
        )
        return int(result.exit_code)
    except QuantCalError as exc:
        payload = {"exit_code": ERROR_EXIT_CODES.get(exc.code, 1), "error": exc.to_dict()}
        if args.report:
            atomic_write_json(
                args.report,
                {
                    "format": "quantcal.fatal-report.v1",
                    "injection_hits": injection_hits or [],
                    **payload,
                },
            )
        _print_json(payload)
        return int(ERROR_EXIT_CODES.get(exc.code, 1))
    except Exception as exc:
        error = QuantCalError(
            "E_INTERNAL", f"{type(exc).__name__}: {exc}", stage="cli"
        )
        payload = {"exit_code": ERROR_EXIT_CODES["E_INTERNAL"], "error": error.to_dict()}
        if args.report:
            atomic_write_json(
                args.report,
                {
                    "format": "quantcal.fatal-report.v1",
                    "injection_hits": injection_hits or [],
                    **payload,
                },
            )
        _print_json(payload)
        return ERROR_EXIT_CODES["E_INTERNAL"]


if __name__ == "__main__":
    raise SystemExit(main())

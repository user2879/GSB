"""Command-line entry point for the column calibration diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cache import write_json_atomic
from .pipeline import run_workbench


def _load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_command(args: argparse.Namespace) -> int:
    left = _load_report(args.left)
    right = _load_report(args.right)
    fields = [
        "metadata",
        "configuration",
        "sample_metadata",
        "injection_hits",
        "columns",
        "sensitivity",
        "failures",
        "summary",
    ]
    mismatches = [field for field in fields if left.get(field) != right.get(field)]
    left_hash = left.get("equivalence", {}).get("deterministic_result_hash")
    right_hash = right.get("equivalence", {}).get("deterministic_result_hash")
    result = {
        "equal": not mismatches and left_hash == right_hash,
        "left": str(args.left),
        "right": str(args.right),
        "left_hash": left_hash,
        "right_hash": right_hash,
        "mismatched_fields": mismatches,
    }
    if args.output:
        write_json_atomic(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["equal"] else 1


def _run_command(args: argparse.Namespace) -> int:
    report = run_workbench(
        config_path=args.config,
        sample_path=args.sample,
        state_dir=args.state_dir,
        report_path=args.report,
        full_rebuild=args.full_rebuild,
    )
    summary = {
        "report": str(args.report),
        "run_version": report["run"]["run_version"],
        "deterministic_result_hash": report["equivalence"][
            "deterministic_result_hash"
        ],
        "complete": report["run"]["complete"],
        "columns": report["summary"]["column_count_completed"],
        "failures": report["summary"]["failure_count"],
        "failures_by_code": report["summary"]["failures_by_code"],
        "reference_passed": report["summary"]["float64_reference"]["passed"],
        "cache_mode": report["cache"]["mode"],
        "reused_columns": len(report["cache"]["reused_columns"]),
        "recomputed_columns": len(report["cache"]["recomputed_columns"]),
        "pending_columns": len(report["cache"]["pending_columns"]),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.fail_on_errors and (report["failures"] or not report["run"]["complete"]):
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcolumn-diag",
        description=(
            "Per-column quantile calibration, fixed-point saturation and "
            "error-attribution diagnostics."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run diagnostics")
    run_parser.add_argument("--config", required=True, help="calibration config JSON")
    run_parser.add_argument("--sample", required=True, help="fixed-seed sample JSON")
    run_parser.add_argument(
        "--state-dir", default="runtime", help="snapshot state directory"
    )
    run_parser.add_argument(
        "--report", default="reports/diagnostic_report.json", help="output report JSON"
    )
    run_parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="ignore persisted column artifacts and rebuild from scratch",
    )
    run_parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="return exit code 2 when the report contains failures or pending work",
    )
    run_parser.set_defaults(func=_run_command)

    compare_parser = subparsers.add_parser(
        "compare", help="compare deterministic report cores"
    )
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", default=None)
    compare_parser.set_defaults(func=_compare_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - user-facing defensive boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

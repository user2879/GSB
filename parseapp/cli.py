"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError
from .pipeline import PipelineError, SimulatedTermination, run_pipeline
from .snapshot import SnapshotError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parseapp",
        description="Deterministic synthetic table/chart layout parser",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="parse all pages in a sample directory")
    run_parser.add_argument("--samples", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--state-dir", type=Path, required=True)
    run_parser.add_argument("--report", type=Path, default=None)
    run_parser.add_argument("--serial", action="store_true", help="use a single worker")
    run_parser.add_argument(
        "--fail-after-commits",
        type=int,
        default=None,
        help="test failpoint: terminate after N atomic snapshot commits",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error("unknown command")
    if args.fail_after_commits is not None and args.fail_after_commits < 1:
        parser.error("--fail-after-commits must be >= 1")
    try:
        report = run_pipeline(
            args.samples,
            args.config,
            args.state_dir,
            serial=args.serial,
            fail_after_commits=args.fail_after_commits,
            report_path=args.report,
        )
    except ConfigError as exc:
        print(json.dumps({"code": exc.code, "message": str(exc)}, ensure_ascii=False))
        return 4
    except SnapshotError as exc:
        print(json.dumps({"code": exc.code, "message": str(exc)}, ensure_ascii=False))
        return 4
    except PipelineError as exc:
        print(json.dumps({"code": exc.code, "message": exc.message}, ensure_ascii=False))
        return 4
    except SimulatedTermination as exc:
        print(json.dumps({"code": "E_SIMULATED_TERMINATION", "message": str(exc)}, ensure_ascii=False))
        return 75

    output_path = args.report if args.report is not None else args.state_dir / "report.json"
    summary = {
        "status": report["status"],
        "counts": report["counts"],
        "snapshot_version": report["snapshot"]["version"],
        "report": str(output_path),
        "serial": args.serial,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if report["status"] == "budget_incomplete":
        return 2
    if report["counts"]["failed"] > 0:
        return 3
    return 0

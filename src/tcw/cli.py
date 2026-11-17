"""Command-line entry point with Windows-compatible integer exit codes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .errors import E_INTERNAL, BusinessError
from .pipeline import build_fatal_report, load_and_execute
from .util import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcw",
        description="Deterministic tabular clustering and density-estimation workbench",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiment.yaml"),
        help="experiment YAML path",
    )
    parser.add_argument("--data", type=Path, default=None, help="override synthetic data description path")
    parser.add_argument(
        "--injections",
        type=Path,
        default=Path("config/injections.yaml"),
        help="independent fault-injection YAML path",
    )
    parser.add_argument("--no-injections", action="store_true", help="run the clean path without an injection config")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("artifacts/snapshots"),
        help="snapshot root directory",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/runs/diagnostic_report.json"),
        help="diagnostic JSON report path",
    )
    parser.add_argument("--snapshot-version", type=int, default=None, help="override snapshot version")
    parser.add_argument("--resume", action="store_true", help="restore and continue matching snapshots")
    return parser


def _print_failures(failures: list[dict[str, object]]) -> None:
    for failure in failures:
        code = failure.get("code", E_INTERNAL)
        stage = failure.get("stage", "unknown")
        injection = failure.get("injection_id")
        suffix = f" injection={injection}" if injection else ""
        print(f"{code} [{stage}]{suffix}: {failure.get('message', '')}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outcome = load_and_execute(
            args.config,
            data_description=args.data,
            injection_path=None if args.no_injections else args.injections,
            snapshot_dir=args.snapshot_dir,
            report_path=args.report,
            resume=args.resume,
            snapshot_version=args.snapshot_version,
        )
    except BusinessError as exc:
        outcome = build_fatal_report(
            exc,
            args.report,
            config_path=args.config,
            injection_path=None if args.no_injections else args.injections,
        )
    except Exception as exc:
        error = BusinessError(
            E_INTERNAL,
            f"{type(exc).__name__}: {exc}",
            stage="internal",
            fatal=True,
        )
        outcome = build_fatal_report(
            error,
            args.report,
            config_path=args.config,
            injection_path=None if args.no_injections else args.injections,
        )

    _print_failures(outcome.failures)
    summary = {
        "status": outcome.report.get("status"),
        "exit_code": outcome.exit_code,
        "report_path": str(args.report.resolve()),
        "failure_count": len(outcome.failures),
        "business_codes": outcome.report.get("business_codes", []),
        "top_candidates": [
            {
                "rank": entry.get("rank"),
                "method": entry.get("method"),
                "params": entry.get("params"),
                "rank_score": entry.get("rank_score"),
            }
            for entry in outcome.report.get("candidate_ranking", [])[:3]
        ],
    }
    print(canonical_json(summary))
    return int(outcome.exit_code)

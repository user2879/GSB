"""Command line interface for the fusion application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .edits import apply_edit_sequence
from .engine import run_workspace
from .samples import generate_samples
from .snapshot import SimulatedCrash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="candidate-fusion", description="Deterministic multi-channel score fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("init-samples", help="generate deterministic local fixtures")
    sample_parser.add_argument("--root", default="samples", help="sample root directory")
    sample_parser.add_argument("--force", action="store_true", help="replace an existing fixture")

    run_parser = subparsers.add_parser("run", help="run one transactional fusion pass")
    run_parser.add_argument("--workspace", required=True)
    run_parser.add_argument("--report", default=None)
    run_parser.add_argument("--inject", default=None, help="independent injection JSON")
    run_parser.add_argument("--mode", choices=["auto", "full"], default="auto")
    run_parser.add_argument("--reset", action="store_true", help="remove committed state before running")
    run_parser.add_argument("--expected-version", type=int, default=None)
    run_parser.add_argument("--simulate-crash-after", choices=["load", "normalize", "fuse", "truncate"], default=None)

    edit_parser = subparsers.add_parser("apply-edits", help="apply a sample edit sequence")
    edit_parser.add_argument("--workspace", required=True)
    edit_parser.add_argument("--sequence", required=True)

    accept_parser = subparsers.add_parser("acceptance", help="run the full local acceptance matrix")
    accept_parser.add_argument("--samples", default="samples")
    accept_parser.add_argument("--output", default="artifacts/acceptance")
    accept_parser.add_argument("--force-fixtures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-samples":
        root = generate_samples(Path(args.root), force=args.force)
        print(json.dumps({"fixture": str(root)}, ensure_ascii=False))
        return 0
    if args.command == "apply-edits":
        applied = apply_edit_sequence(Path(args.workspace), Path(args.sequence))
        print(json.dumps({"applied": applied}, ensure_ascii=False))
        return 0
    if args.command == "run":
        report_path = Path(args.report) if args.report else Path(args.workspace) / "report.json"
        try:
            outcome = run_workspace(
                Path(args.workspace),
                report_path=report_path,
                injection_path=Path(args.inject) if args.inject else None,
                mode=args.mode,
                reset=args.reset,
                expected_version=args.expected_version,
                simulate_crash_after=args.simulate_crash_after,
            )
        except SimulatedCrash as exc:
            print(str(exc), file=sys.stderr)
            return 75
        report = outcome.report
        summary = {
            "failures": len(report.get("failures") or []),
            "kept_count": (report.get("fusion_summary") or {}).get("kept_count"),
            "mode": report.get("mode"),
            "run_version": report.get("run_version"),
            "snapshot_stale": (report.get("snapshot") or {}).get("stale"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    if args.command == "acceptance":
        from .acceptance import run_acceptance

        summary = run_acceptance(
            Path(args.samples),
            Path(args.output),
            force_fixtures=args.force_fixtures,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary.get("status") == "passed" else 1
    return 2

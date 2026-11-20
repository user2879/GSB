"""Command-line entry point for sample generation and workbench runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run_workbench
from .samplegen import generate_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aedwb",
        description="AED shard calibration and confidence-drift diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic offline samples")
    generate.add_argument("--out", required=True, help="sample directory to create")
    generate.add_argument(
        "--injected",
        action="store_true",
        help="add the five isolation/fault-injection probe packets",
    )
    generate.add_argument(
        "--no-clean",
        action="store_true",
        help="do not delete an existing output directory first",
    )

    run = subparsers.add_parser("run", help="run the workbench on a sample directory")
    run.add_argument("--sample-dir", required=True, help="local sample directory")
    run.add_argument("--state-dir", default=None, help="cache/snapshot directory")
    run.add_argument("--report", default=None, help="JSON diagnostic report path")
    run.add_argument(
        "--summary",
        action="store_true",
        help="print a compact run summary instead of the full report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        path = generate_sample(args.out, injected=args.injected, clean=not args.no_clean)
        print(f"generated: {path}")
        return 0

    sample_dir = Path(args.sample_dir)
    state_dir = Path(args.state_dir) if args.state_dir else sample_dir / ".workbench"
    report_path = Path(args.report) if args.report else state_dir / "report.json"
    try:
        report = run_workbench(sample_dir, state_dir, report_path)
    except Exception as error:  # configuration/input setup failure
        print(json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False))
        return 2

    if args.summary:
        summary = {
            "report": str(report_path),
            "consistency_signature": report["consistency_signature"],
            "config_hash": report["run"]["config_hash"],
            "calibration_fingerprint": report["run"]["calibration_fingerprint"],
            "snapshot_status": report["run"]["snapshot_status"]["status"],
            "snapshot_error": report["run"]["snapshot_status"]["error_code"],
            "packet_count": len(report["packets"]),
            "valid_packet_count": sum(1 for packet in report["packets"] if packet["valid"]),
            "global_frame_count": len(report["global_frames"]),
            "drift_alert_count": len(report["drift_alerts"]),
            "boundary_sample_count": len(report["boundary_samples"]),
            "failure_count": len(report["failures"]),
            "injection_hit_count": sum(1 for item in report["injection_hits"] if item["hit"]),
            "cache_summary": report["ledger"]["cache_summary"],
            "elapsed_ms": round(report["run"]["elapsed_ms"], 3),
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

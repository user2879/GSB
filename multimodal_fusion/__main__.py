"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import FusionError
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic fixed-point multimodal fusion")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="run the fusion pipeline")
    run.add_argument("--config", required=True, help="fusion YAML path")
    run.add_argument("--input", required=True, help="input sample directory")
    run.add_argument("--state", default=".fusion_state", help="snapshot state directory")
    run.add_argument("--report", default="reports/latest.json", help="JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command != "run":
        parser.print_help()
        return 2
    try:
        report = run_pipeline(
            config_path=arguments.config,
            input_dir=arguments.input,
            state_dir=arguments.state,
            report_path=arguments.report,
        )
    except FusionError as exc:
        error_report = {"code": exc.code, "message": exc.message, "details": exc.details}
        report_path = Path(arguments.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(error_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "global_version": report["global_version"],
                "scored_samples": report["summary"]["scored_samples"],
                "pending_samples": report["summary"]["pending_samples"],
                "failure_count": len(report["failure_list"]),
                "report": arguments.report,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

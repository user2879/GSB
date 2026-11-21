"""CLI entry point for the deterministic clustering diagnostics workbench."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from clustering_workbench.pipeline import WorkbenchPipeline
from clustering_workbench.server import WorkbenchServer, verify_serial_parallel
from clustering_workbench.utils import json_safe


def _print_report_summary(report: dict) -> None:
    summary = {
        "status": report.get("run", {}).get("status"),
        "rows": report.get("dataset", {}).get("rows_generated"),
        "preprocess": {
            "method": report.get("preprocess", {}).get("method"),
            "shape": report.get("preprocess", {}).get("shape"),
            "source": report.get("preprocess", {}).get("source"),
            "content_fingerprint": report.get("preprocess", {}).get("content_fingerprint"),
        },
        "ranking": [
            {
                "rank": row.get("rank"),
                "method": row.get("method"),
                "metrics": row.get("metrics"),
                "flags": row.get("diagnostic_flags"),
            }
            for row in report.get("ranking", [])
        ],
        "budget": {
            "reserved": report.get("scheduler", {}).get("budget_reserved"),
            "executed": report.get("scheduler", {}).get("iterations_executed_total"),
        },
        "reuse": report.get("reuse", {}).get("status_counts"),
        "injections": {
            "hit": report.get("injections", {}).get("hit"),
            "missed": report.get("injections", {}).get("missed"),
        },
        "failures": sorted({failure.get("code") for failure in report.get("failures", [])}),
        "determinism_digest": report.get("determinism_digest"),
        "output": report.get("run", {}).get("output_dir"),
    }
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic clustering diagnostics workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the configured experiment")
    run_parser.add_argument("--config", default="configs/demo.yaml")
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument("--json", action="store_true", help="Print the full JSON report")

    verify_parser = subparsers.add_parser("verify", help="Compare serial and parallel execution")
    verify_parser.add_argument("--config", default="configs/demo.yaml")
    verify_parser.add_argument("--json", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Start the local interactive console")
    serve_parser.add_argument("--config", default="configs/demo.yaml")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--open", action="store_true")

    args = parser.parse_args()
    if args.command == "run":
        report = WorkbenchPipeline(args.config, output_dir=args.output).run()
        if args.json:
            print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
        else:
            _print_report_summary(report)
    elif args.command == "verify":
        result = verify_serial_parallel(args.config, {})
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                f"digest_equal={result['digest_equal']} ranking_equal={result['ranking_equal']} "
                f"labels_equal={result['labels_equal']} budget_equal={result['budget_equal']} "
                f"failures_equal={result['failures_equal']}"
            )
            print(f"serial_digest={result['serial_digest']}")
            print(f"parallel_digest={result['parallel_digest']}")
    elif args.command == "serve":
        if args.open:
            webbrowser.open(f"http://{args.host}:{args.port}")
        WorkbenchServer(args.config, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()

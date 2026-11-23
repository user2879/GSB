from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import DiagnosticRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rope-diag",
        description="Deterministic RoPE extrapolation diagnostics for long-context attention.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("sample_data/config.json"), help="position encoding JSON"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("sample_data/samples.jsonl"),
        help="attention score sample JSONL",
    )
    parser.add_argument(
        "--pruning",
        type=Path,
        default=Path("sample_data/pruning_strategies.json"),
        help="dimension pruning policy table",
    )
    parser.add_argument(
        "--injections",
        type=Path,
        default=Path("sample_data/injections.json"),
        help="fault injection manifest",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="workspace root; snapshots and temp directories stay below this path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report JSON path (default: <workspace>/artifacts/latest/report.json)",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="ignore all cache hits and recompute every selected stage",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the compact JSON summary",
    )
    return parser


def _summary(report: dict) -> dict:
    return {
        "report_schema": report.get("schema_version"),
        "semantic_digest": report.get("semantic_digest"),
        "config_hash": report.get("inputs", {}).get("config_hash", ""),
        "validation": {
            "config_valid": report.get("validation", {}).get("config_valid"),
            "sample_valid_count": report.get("validation", {}).get("sample_valid_count"),
            "sample_invalid_count": report.get("validation", {}).get("sample_invalid_count"),
        },
        "strategy": {
            "strategy_id": (report.get("strategy") or {}).get("strategy_id"),
            "dimension_count": (report.get("strategy") or {}).get("dimension_count"),
            "scoring_chunk_size": (report.get("strategy") or {}).get("scoring_chunk_size"),
        },
        "schemes": {
            scheme: {
                "status": payload.get("status"),
                "completed": payload.get("sample_summary", {}).get("completed"),
                "planned": payload.get("sample_summary", {}).get("planned"),
                "extrapolation_scores": payload.get("statistics", {})
                .get("extrapolation", {})
                .get("score_count"),
                "mean_score_deviation": next(
                    (
                        item.get("mean_deviation")
                        for item in payload.get("trace", {}).get("stages", [])
                        if item.get("stage") == "score"
                    ),
                    None,
                ),
            }
            for scheme, payload in report.get("schemes", {}).items()
        },
        "scheduler": {
            "approved": report.get("scheduler", {}).get("approved_count"),
            "reused": report.get("scheduler", {}).get("reused_count"),
            "pending": report.get("scheduler", {}).get("pending_count"),
            "remaining_budget": report.get("scheduler", {}).get("remaining_budget"),
        },
        "cache": {
            "reused_stage_count": len(report.get("cache", {}).get("reused", [])),
            "invalidated_stage_count": len(report.get("cache", {}).get("invalidated", [])),
        },
        "injections_triggered": sorted(
            {
                item["injection_id"]
                for item in report.get("injections", {}).get("triggered", [])
            }
        ),
        "failure_count": len(report.get("failures", [])),
        "failure_codes": sorted({item["code"] for item in report.get("failures", [])}),
        "output": report.get("run_metadata", {}).get("output_path"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = DiagnosticRunner(
        config_path=args.config,
        samples_path=args.samples,
        pruning_path=args.pruning,
        injections_path=args.injections,
        workspace=args.workspace,
        output_path=args.output,
        force_full=args.force_full,
    )
    report = runner.run()
    summary = _summary(report)
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if not report.get("validation", {}).get("config_valid", False):
        return 2
    if report.get("scheduler", {}).get("pending_count", 0):
        return 3
    completed_schemes = [
        payload
        for payload in report.get("schemes", {}).values()
        if payload.get("sample_summary", {}).get("completed", 0) > 0
    ]
    if not completed_schemes:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

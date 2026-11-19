"""Command-line entry point for the diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_experiment
from .errors import ConfigLoadError, E_CONFIG_INVALID
from .evaluation import run_experiment
from .serialization import atomic_write_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="feature-diag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run all valid scenarios and write reports")
    run_parser.add_argument("--config", required=True, help="path to experiment YAML")
    run_parser.add_argument("--out", default="artifacts/reports/latest", help="report output directory")
    run_parser.add_argument("--cache-dir", default=None, help="override content cache directory")
    run_parser.add_argument("--snapshot-dir", default=None, help="override snapshot root directory")
    run_parser.add_argument("--max-workers", type=int, default=None, help="cap concurrent workers (never above config)")
    run_parser.add_argument(
        "--stop-after-stage",
        choices=("crossing", "reductions", "metrics", "diagnostics"),
        default=None,
        help="commit a stage and stop, for recovery testing",
    )
    run_parser.add_argument("--no-cache", action="store_true", help="disable content artifact cache")

    validate_parser = subparsers.add_parser("validate", help="load and validate configuration without running")
    validate_parser.add_argument("--config", required=True, help="path to experiment YAML")
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _handle_load_error(config_path: str, error: ConfigLoadError) -> int:
    failure = error.as_failure().as_dict()
    out_dir = Path("artifacts/reports/config-error")
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_dir / "report.json",
        json.dumps(
            {
                "status": "failed",
                "config": config_path,
                "code": E_CONFIG_INVALID,
                "failures": [failure],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    _print_json({"status": "failed", "code": E_CONFIG_INVALID, "failures": [failure]})
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        experiment = load_experiment(args.config)
    except ConfigLoadError as exc:
        return _handle_load_error(args.config, exc)

    if args.command == "validate":
        payload = {
            "status": "valid" if not experiment.failures else "partial",
            "config_hash": experiment.config_hash,
            "valid_scenarios": [scenario.scenario_id for scenario in experiment.scenarios],
            "failures": [failure.as_dict() for failure in experiment.failures],
            "precision_fingerprints": experiment.precision_fingerprints,
        }
        _print_json(payload)
        return 0

    result = run_experiment(
        experiment,
        output_dir=args.out,
        cache_dir=args.cache_dir,
        snapshot_dir=args.snapshot_dir,
        max_workers=args.max_workers,
        stop_after_stage=args.stop_after_stage,
        cache_enabled=not args.no_cache,
    )
    if result.partial:
        payload = {
            "status": "partial",
            "stopped_after_stage": result.stopped_after_stage,
            "config_hash": experiment.config_hash,
            "outputs": result.report_paths,
            "snapshot": result.snapshot_info,
            "failures": [failure.as_dict() for failure in experiment.failures],
        }
        _print_json(payload)
        return 75

    assert result.report is not None
    payload = {
        "status": "ok",
        "config_hash": experiment.config_hash,
        "outputs": result.report_paths,
        "summary": result.report["summary"],
        "analysis_fingerprint": result.report["analysis_fingerprint"],
        "report_fingerprint": result.report["report_fingerprint"],
        "snapshot": {
            "resumed_stages": result.snapshot_info.get("resumed_stages"),
            "computed_stages": result.snapshot_info.get("computed_stages"),
            "recovery_point": result.snapshot_info.get("recovery_point"),
            "stale_event": result.snapshot_info.get("stale_event"),
        },
        "failures": result.report["failures"],
    }
    _print_json(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

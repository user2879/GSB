"""Command-line entry points for preparation, execution, and replacement data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .pipeline import run_pipeline
from .sample import prepare_sample, read_replacement_file, write_column_override


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _config(args: argparse.Namespace):
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    return load_pipeline_config(config_path, root=root)


def _print_failures(failures: list[dict[str, Any]]) -> None:
    for item in failures:
        print(
            f"{item['code']} [{item['item']}] {item['message']}",
            file=sys.stderr,
        )


def command_prepare(args: argparse.Namespace) -> int:
    config = _config(args)
    if config.failures:
        _print_failures(config.failures)
        return 2
    manifest = prepare_sample(config, force=args.force)
    _json_dump(manifest)
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _config(args)
    report = run_pipeline(config)
    summary = {
        "status": report["status"],
        "report": report.get("artifacts", {}).get("report"),
        "config_hash": report.get("config_hash"),
        "sample_checksum": report.get("sample_checksum"),
        "effective_data_hash": report.get("effective_data_hash"),
        "global_version": report.get("global_version"),
        "stage_versions": report.get("stage_versions"),
        "failures": report.get("failures", []),
    }
    if report["status"] == "ok":
        summary.update(
            {
                "cross_feature_count": report["cross_encoding"]["active_feature_count"],
                "degradation_count": len(report["cross_encoding"]["degradations"]),
                "paths": {
                    item["path"]: {
                        "retained_dimensions": item["retained_dimensions"],
                        "summary": item["summary"],
                    }
                    for item in report["reduction"]["paths"]
                },
                "first_exceeded_stage": report["diagnostics"]["first_exceeded_stage"],
                "boundary_case_count": len(report["diagnostics"]["boundary_cases"]),
                "cache_summary": report["diagnostics"]["cache_summary"],
                "max_observed_workers": report["runtime"]["max_observed_workers"],
            }
        )
    _json_dump(summary)
    return 0 if report["status"] == "ok" else 2


def command_replace_column(args: argparse.Namespace) -> int:
    config = _config(args)
    if config.failures:
        _print_failures(config.failures)
        return 2
    try:
        values = read_replacement_file(args.values_file)
        result = write_column_override(config, args.name, values)
    except Exception as exc:
        print(f"replacement failed: {exc}", file=sys.stderr)
        return 2
    _json_dump({"status": "ok", "replacement": result})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = _config(args)
    payload = {
        "status": "ok" if not config.failures else "failed",
        "config_hash": config.config_hash,
        "loaded_columns": [descriptor["name"] for descriptor in config.descriptors],
        "rows": config.rows,
        "failures": config.failures,
    }
    _json_dump(payload)
    return 0 if not config.failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-pipeline",
        description="Deterministic feature crossing, reduction, precision, and cache auditor",
    )
    parser.add_argument(
        "--root",
        default=str(Path.cwd()),
        help="workspace root used to resolve relative paths (default: current directory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="generate or verify the persisted deterministic sample")
    prepare_parser.add_argument("--config", required=True, help="pipeline YAML path")
    prepare_parser.add_argument("--force", action="store_true", help="regenerate the base sample")
    prepare_parser.set_defaults(func=command_prepare)

    run_parser = subparsers.add_parser("run", help="execute the complete pipeline")
    run_parser.add_argument("--config", required=True, help="pipeline YAML path")
    run_parser.set_defaults(func=command_run)

    replace_parser = subparsers.add_parser("replace-column", help="atomically persist a single-column replacement")
    replace_parser.add_argument("--config", required=True, help="pipeline YAML path")
    replace_parser.add_argument("--name", required=True, help="column to replace")
    replace_parser.add_argument("--values-file", required=True, help="YAML or JSON list / values mapping")
    replace_parser.set_defaults(func=command_replace_column)

    validate_parser = subparsers.add_parser("validate", help="validate configuration and collect business codes")
    validate_parser.add_argument("--config", required=True, help="pipeline YAML path")
    validate_parser.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

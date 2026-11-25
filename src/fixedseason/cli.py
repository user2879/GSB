"""Command-line interface for the fixedseason experiment bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .diagnostics import run_width_sign_check
from .engine import FitEngine
from .errors import Issue
from .experiment import (
    clear_generated_artifacts,
    compare_reports,
    run_fresh_build,
    run_incremental_experiment,
    workspace_generated_entries,
)
from .jsonutil import pretty_json, write_json_atomic
from .loader import load_config, load_series_jsonl
from .synthetic import generate_example_workspace
from .verification import run_verification


def _failure_report(
    *,
    workspace: Path,
    config_path: Path,
    issues: list[Issue],
    mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "global_version": None,
            "config_hash": None,
            "compute_config_hash": None,
            "bit_width_tier": None,
            "mode": mode,
            "snapshot_status": "not_created",
            "resumed": False,
            "interrupted": False,
            "complete": False,
        },
        "config": None,
        "validation_failures": [issue.as_dict() for issue in issues],
        "fit_failures": [],
        "failures": [issue.as_dict() for issue in issues],
        "pending": [],
        "unfinished": [],
        "series": [],
        "cache_ledger": [],
        "reuse_ledger": [],
        "budget": {
            "settings": None,
            "consumed_fit_points": 0,
            "remaining_fit_points": 0,
            "ledger": [],
            "batches": [],
            "pending_queue": [],
            "cache_hit_series": [],
        },
        "summary": {
            "input_series_count": 0,
            "completed_series_count": 0,
            "validation_failure_count": len(issues),
            "fit_failure_count": 0,
            "pending_count": 0,
        },
        "input": {"config": str(config_path), "workspace": str(workspace)},
    }


def _load_inputs(series_path: Path, config_path: Path):
    series, series_issues = load_series_jsonl(series_path)
    config, config_issues = load_config(config_path)
    return series, series_issues, config, config_issues


def _summary_text(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    run = report.get("run", {})
    lines = [
        f"complete={run.get('complete')} version={run.get('global_version')} "
        f"mode={run.get('mode')} snapshot={run.get('snapshot_status')}",
        f"series={summary.get('completed_series_count', 0)} "
        f"cache_hits={summary.get('cache_hit_series_count', 0)} "
        f"pending={summary.get('pending_count', 0)} "
        f"failures={len(report.get('failures', []))}",
    ]
    if report.get("failures"):
        for issue in report["failures"][:5]:
            lines.append(f"  {issue.get('code')}: {issue.get('message')}")
    if report.get("pending"):
        for issue in report["pending"][:5]:
            lines.append(f"  {issue.get('code')}: {issue.get('message')}")
    return "\n".join(lines)


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_example_workspace(args.directory, force=args.force)
    print(pretty_json(paths), end="")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    existing = workspace_generated_entries(workspace)
    if args.fresh:
        clear_generated_artifacts(workspace)
    elif existing:
        print(
            "existing generated entries detected: "
            + ", ".join(existing)
            + " (use --fresh for a blank run)",
            file=sys.stderr,
        )
    series_path = Path(args.series)
    config_path = Path(args.config)
    series, series_issues, config, config_issues = _load_inputs(series_path, config_path)
    mode = "serial" if args.serial else "concurrent"
    if config is None or config_issues:
        report = _failure_report(
            workspace=workspace,
            config_path=config_path,
            issues=series_issues + config_issues,
            mode=mode,
        )
        write_json_atomic(workspace / "final_report.json", report)
        print(_summary_text(report), file=sys.stderr)
        return 2

    if args.commands and args.stop_after is not None:
        print("--stop-after can only be used without --commands", file=sys.stderr)
        return 2
    equivalence: dict[str, Any] | None = None
    if args.commands:
        outcome = run_incremental_experiment(
            workspace,
            series,
            config,
            initial_issues=series_issues,
            commands_path=args.commands,
            mode=mode,
            fresh=False,
            save_reports=True,
        )
        final_report = outcome.final_report
        write_json_atomic(
            workspace / "incremental-trace.json",
            {
                "schema_version": 1,
                "steps": outcome.trace,
                "issues": [issue.as_dict() for issue in outcome.issues],
                "final_version": outcome.final_config.version,
            },
        )
        if args.check_equivalence:
            fresh_report = run_fresh_build(
                workspace / "equivalence-full",
                outcome.final_series,
                outcome.final_config,
                validation_issues=outcome.issues,
                mode=mode,
            )
            equal, differences = compare_reports(final_report, fresh_report)
            equivalence = {
                "schema_version": 1,
                "equal": equal,
                "differences": differences[:100],
                "incremental_final_version": outcome.final_config.version,
            }
            write_json_atomic(workspace / "equivalence-report.json", equivalence)
            final_report = dict(final_report)
            final_report["equivalence"] = equivalence
    else:
        engine = FitEngine(
            workspace,
            config,
            series,
            validation_issues=series_issues,
            mode=mode,
        )
        final_report = engine.run(
            version=config.version,
            stop_after=args.stop_after,
            force_rerun=args.force_rerun,
        )
    write_json_atomic(workspace / "final_report.json", final_report)
    if args.json:
        print(pretty_json(final_report), end="")
    else:
        print(_summary_text(final_report))
        print(f"report: {workspace / 'final_report.json'}")
    if equivalence is not None and not equivalence["equal"]:
        return 3
    return 0


def _cmd_width_check(args: argparse.Namespace) -> int:
    series, series_issues = load_series_jsonl(args.series)
    config, config_issues = load_config(args.config)
    if series_issues:
        print(f"{len(series_issues)} series validation issue(s) recorded", file=sys.stderr)
    if config is None or config_issues:
        for issue in config_issues:
            print(f"{issue.code}: {issue.message}", file=sys.stderr)
        return 2
    widths = args.bits or list(config.comparison_bit_widths) or [32, 48]
    try:
        report = run_width_sign_check(
            args.workspace,
            series,
            config,
            widths,
            mode="serial" if args.serial else "concurrent",
        )
    except ValueError as exc:
        print(f"width-check error: {exc}", file=sys.stderr)
        return 2
    output = Path(args.workspace) / "width-sign-report.json"
    write_json_atomic(output, report)
    if args.json:
        print(pretty_json(report), end="")
    else:
        print(
            f"widths={report['widths']} comparisons={len(report['series_comparisons'])} "
            f"flipped={report['flipped_series']}"
        )
        print(f"report: {output}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = run_verification(
        args.workspace,
        fixture_dir=args.fixtures,
        fresh=not args.keep,
    )
    if args.json:
        print(pretty_json(report), end="")
    else:
        for name, check in report["checks"].items():
            print(f"{'PASS' if check.get('passed') else 'FAIL'}  {name}")
        print(f"overall_passed={report['overall_passed']}")
        print(f"report: {Path(args.workspace) / 'verification-report.json'}")
    return 0 if report["overall_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fixedseason",
        description="Deterministic fixed-point seasonal fitting experiment bench",
    )
    parser.add_argument("--version", action="version", version="fixedseason 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write deterministic demo fixtures")
    generate.add_argument("--directory", default="examples")
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=_cmd_generate)

    run = subparsers.add_parser("run", help="fit a series library")
    run.add_argument("--series", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--commands")
    run.add_argument("--workspace", required=True)
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--serial", action="store_true")
    run.add_argument("--stop-after", type=int)
    run.add_argument("--force-rerun", action="store_true")
    run.add_argument(
        "--check-equivalence",
        action="store_true",
        help="after commands, compare final state with a blank full rebuild",
    )
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=_cmd_run)

    widths = subparsers.add_parser(
        "width-check", help="compare trend-slope signs across two fixed-point widths"
    )
    widths.add_argument("--series", required=True)
    widths.add_argument("--config", required=True)
    widths.add_argument("--workspace", required=True)
    widths.add_argument("--bits", type=int, nargs=2)
    widths.add_argument("--serial", action="store_true")
    widths.add_argument("--json", action="store_true")
    widths.set_defaults(func=_cmd_width_check)

    verify = subparsers.add_parser(
        "verify", help="run acceptance checks and write machine-readable evidence"
    )
    verify.add_argument("--workspace", default="artifacts/verification")
    verify.add_argument("--fixtures")
    verify.add_argument("--keep", action="store_true")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

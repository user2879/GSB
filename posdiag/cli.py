from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from .models import ConfigValidationError, load_config, load_edit_plan, load_samples
from .pipeline import build_failure_report, run_pipeline
from .samples import generate_sample_directory


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _parse_scheme_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    report_path = Path(args.report)
    try:
        config = load_config(args.config)
    except ConfigValidationError as exc:
        build_failure_report(exc.issues, workspace=workspace, report_path=report_path)
        _print_json(
            {
                "status": "failed",
                "report": str(report_path),
                "diagnostic_codes": sorted({issue.code for issue in exc.issues} | {"E_CONFIG_INVALID"}),
                "compliance_errors": [issue.as_dict() for issue in exc.issues],
                "config_errors": [issue.as_dict() for issue in exc.issues],
            }
        )
        return 2

    loaded_samples = load_samples(args.samples, dimension=config.dimension)
    edited = load_edit_plan(config, loaded_samples.samples, args.edit_plan)

    schemes_override: Sequence[str] | None = None
    if args.all_schemes:
        schemes_override = ["rope_linear", "rope_ntk", "rope_standard"]
    elif args.schemes:
        schemes_override = _parse_scheme_list(args.schemes)

    stop_event = threading.Event()
    received_signal: list[int] = []

    def handle_signal(signum: int, _frame: object) -> None:
        received_signal.append(signum)
        stop_event.set()
        print(
            f"received signal {signum}; finishing the current stage before exit",
            file=sys.stderr,
            flush=True,
        )

    previous_handlers: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    def stage_callback(stage: str) -> None:
        if args.stop_after_stage and stage == args.stop_after_stage:
            stop_event.set()

    try:
        run = run_pipeline(
            edited.config,
            edited.samples,
            workspace,
            sample_issues=loaded_samples.issues,
            edit_issues=edited.issues,
            edit_plan_fingerprint=edited.plan_fingerprint,
            applied_edit_operations=edited.applied_operations,
            report_path=report_path,
            stop_event=stop_event,
            stage_complete_callback=stage_callback,
            schemes_override=schemes_override,
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    metadata = run.report["run_metadata"]
    summary = {
        "status": run.report["status"],
        "report": str(report_path),
        "config_hash": run.report["config_hash"],
        "summary": run.report["summary"],
        "diagnostic_codes": metadata["diagnostic_codes"],
        "resumed": metadata["resumed"],
        "stale": metadata["stale"],
        "cache": metadata["cache"],
        "snapshot_manifest": str(workspace / "manifest.json"),
        "over_tolerance_samples": [
            {"sample_id": item["sample_id"], "scheme": item["scheme"], "stage": item["first_exceeded"]}
            for item in run.report["over_tolerance_samples"]
        ],
        "budget_exhausted": run.report["budget_usage"].get("budget_exhausted_samples", []),
        "pending_queue": run.report["budget_usage"].get("pending_queue", []),
    }
    _print_json(summary)
    if received_signal:
        return 130
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posdiag", description="Long-context positional-encoding diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the diagnostic pipeline")
    run_parser.add_argument("--config", required=True, help="position encoding config JSON")
    run_parser.add_argument("--samples", required=True, help="attention sample JSONL")
    run_parser.add_argument("--edit-plan", help="optional edit plan JSON")
    run_parser.add_argument("--workspace", default=".posdiag_state", help="snapshot/workspace directory")
    run_parser.add_argument("--report", default="artifacts/diagnostic_report.json", help="output report JSON")
    schemes = run_parser.add_mutually_exclusive_group()
    schemes.add_argument("--all-schemes", action="store_true", help="run standard, linear and NTK-aware schemes")
    schemes.add_argument("--schemes", help="comma separated scheme names to run")
    run_parser.add_argument(
        "--stop-after-stage",
        choices=("loading", "encoding", "attention", "error_trace"),
        help="deterministic graceful-stop hook; useful for snapshot recovery tests",
    )
    run_parser.set_defaults(handler=_run_command)

    generate_parser = subparsers.add_parser("generate", help="generate deterministic sample inputs")
    generate_parser.add_argument("--output", default="samples")
    generate_parser.add_argument("--seed", type=int, default=20261124)

    validate_parser = subparsers.add_parser("validate", help="validate local inputs without running diagnostics")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--samples", required=True)
    validate_parser.add_argument("--edit-plan")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return args.handler(args)
    if args.command == "generate":
        generated = generate_sample_directory(args.output, seed=args.seed)
        _print_json({name: str(path) for name, path in generated.items()})
        return 0
    if args.command == "validate":
        try:
            config = load_config(args.config)
            loaded = load_samples(args.samples, dimension=config.dimension)
            edited = load_edit_plan(config, loaded.samples, args.edit_plan)
        except ConfigValidationError as exc:
            _print_json({"valid": False, "config_errors": [issue.as_dict() for issue in exc.issues]})
            return 2
        issues = [issue.as_dict() for issue in loaded.issues + edited.issues]
        _print_json(
            {
                "valid": not issues,
                "valid_samples": len(loaded.samples),
                "config_hash": edited.config.fingerprint(),
                "sample_errors": [issue.as_dict() for issue in loaded.issues],
                "edit_errors": [issue.as_dict() for issue in edited.issues],
            }
        )
        return 0 if not issues else 1
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry points for incremental runs and equivalence verification."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from .engine import Engine
from .report import write_json_report
from .verify import run_verification, scenario_batches


def _load_inputs(config_path: str, spec_path: str, edits_path: str | None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec_data = json.load(handle)
    batches: list[dict[str, Any]] = []
    if edits_path:
        with open(edits_path, "r", encoding="utf-8") as handle:
            batches = json.load(handle)
        if not isinstance(batches, list):
            raise ValueError("edit plan must be a JSON list")
    return config_data, spec_data, batches


def _parse_faults(values: list[str]) -> dict[int, str]:
    faults: dict[int, str] = {}
    allowed = {"sampling", "split", "leaf"}
    for value in values:
        try:
            tree_text, stage = value.split(":", 1)
            tree_index = int(tree_text)
        except ValueError as exception:
            raise ValueError(f"invalid fault {value!r}; expected TREE_INDEX:sampling|split|leaf") from exception
        if tree_index < 0 or stage not in allowed:
            raise ValueError(f"invalid fault {value!r}; expected TREE_INDEX:sampling|split|leaf")
        faults[tree_index] = stage
    return faults


def _install_stop_handler(event: threading.Event) -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        if not event.is_set():
            print(f"received signal {signum}; will stop after in-flight trees and commit a snapshot", file=sys.stderr)
        event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def _safe_label(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "-" for character in name).strip("-") or "batch"


def command_run(args: argparse.Namespace) -> int:
    config_data, spec_data, batches = _load_inputs(args.config, args.data_spec, args.edits)
    faults = _parse_faults(args.inject_fault)
    state_dir = Path(args.state_dir)
    engine = Engine(state_dir, tmp_dir=args.tmp_dir, report_dir=args.report_dir)
    stop_event = threading.Event()
    _install_stop_handler(stop_event)

    through = args.through_batch
    scenarios = scenario_batches(batches, through)
    last_result = None
    for index, (name, cumulative) in enumerate(scenarios):
        report_name = f"report-{index:02d}-{_safe_label(name)}.json"
        result = engine.run(
            config_data,
            spec_data,
            cumulative,
            label=name,
            report_name=report_name,
            faults=faults,
            stop_event=stop_event,
            tree_delay_ms=args.tree_delay_ms,
        )
        last_result = result
        write_json_report(Path(args.report_dir) / "latest.json", result.report)
        print(
            json.dumps(
                {
                    "scenario": name,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "report": result.report_path,
                    "reused": result.cache_audit.get("reused", []),
                    "recomputed": result.cache_audit.get("recomputed", []),
                    "failures": result.failures,
                    "metrics": result.artifacts.get("ensemble", {}).get("metrics"),
                },
                ensure_ascii=False,
            )
        )
        if result.status != "completed":
            return result.exit_code
    return last_result.exit_code if last_result is not None else 0


def command_verify(args: argparse.Namespace) -> int:
    config_data, spec_data, batches = _load_inputs(args.config, args.data_spec, args.edits)
    faults = _parse_faults(args.inject_fault)
    verification, exit_code = run_verification(
        config_data=config_data,
        spec_data=spec_data,
        batches=batches,
        work_dir=args.work_dir,
        through_batch=args.through_batch,
        parallel_workers=args.parallel_workers,
        faults=faults,
        tree_delay_ms=args.tree_delay_ms,
    )
    print(
        json.dumps(
            {
                "all_equal": verification["all_equal"],
                "scenario_count": verification["scenario_count"],
                "concurrency_equal": verification.get("concurrency", {}).get("equal") if verification.get("concurrency") else None,
                "verification": str(Path(verification["root"]) / "verification.json"),
            },
            ensure_ascii=False,
        )
    )
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental tree-ensemble workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="apply an edit plan incrementally and write per-scenario reports")
    run.add_argument("--config", required=True)
    run.add_argument("--data-spec", required=True)
    run.add_argument("--edits")
    run.add_argument("--state-dir", default="work/state")
    run.add_argument("--tmp-dir", default="work/tmp")
    run.add_argument("--report-dir", default="work/reports")
    run.add_argument("--through-batch", type=int, default=None, help="apply only the first N edit batches")
    run.add_argument("--inject-fault", action="append", default=[], metavar="TREE:STAGE")
    run.add_argument("--tree-delay-ms", type=float, default=0.0, help="test hook for graceful-signal validation")
    run.set_defaults(func=command_run)

    verify = subparsers.add_parser("verify", help="compare every incremental batch with an empty-state full rebuild")
    verify.add_argument("--config", required=True)
    verify.add_argument("--data-spec", required=True)
    verify.add_argument("--edits")
    verify.add_argument("--work-dir", default="work")
    verify.add_argument("--through-batch", type=int, default=None)
    verify.add_argument("--parallel-workers", type=int, default=4)
    verify.add_argument("--inject-fault", action="append", default=[], metavar="TREE:STAGE")
    verify.add_argument("--tree-delay-ms", type=float, default=0.0)
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, read_json
from .compare import compare_reports
from .demo import run_demo
from .generator import generate_fixture
from .pipeline import PipelineOptions, run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ts_pipeline",
        description="Deterministic local time-series backtest and verification pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic fixtures")
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=20261122)
    generate.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="run the staged pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--state-dir", type=Path)
    run.add_argument("--report", type=Path)
    run.add_argument("--samples", type=Path)
    run.add_argument("--injections", type=Path)
    run.add_argument("--reset", action="store_true", help="remove state before running")
    run.add_argument(
        "--stop-after-stage",
        choices=["load", "windows", "models", "metrics"],
        help="test hook: atomically commit the named stage and exit 0",
    )

    compare = subparsers.add_parser("compare", help="compare two JSON reports")
    compare.add_argument("--left", required=True, type=Path)
    compare.add_argument("--right", required=True, type=Path)
    compare.add_argument("--out", type=Path)

    demo = subparsers.add_parser("demo", help="run all acceptance verification scenarios")
    demo.add_argument("--root", type=Path, default=Path("demo_workspace"))
    demo.add_argument("--seed", type=int, default=20261122)
    demo.add_argument("--force", action="store_true")
    return parser


def _install_signal_handlers(stop_event: threading.Event) -> list[tuple[int, Any]]:
    installed: list[tuple[int, Any]] = []

    def handler(signum: int, frame: Any) -> None:
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous = signal.signal(signum, handler)
        except (OSError, ValueError):
            continue
        installed.append((signum, previous))
    return installed


def _restore_signal_handlers(installed: list[tuple[int, Any]]) -> None:
    for signum, previous in installed:
        try:
            signal.signal(signum, previous)
        except (OSError, ValueError):
            pass


def _cmd_run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    state_dir = (args.state_dir or (config_path.parent / ".pipeline_state")).resolve()
    report_path = (args.report or (config_path.parent / "report.json")).resolve()
    if args.reset and state_dir.exists():
        shutil.rmtree(state_dir)
    stop_event = threading.Event()
    installed = _install_signal_handlers(stop_event)
    try:
        result = run_pipeline(
            PipelineOptions(
                config_path=config_path,
                state_dir=state_dir,
                report_path=report_path,
                samples_override=args.samples.resolve() if args.samples else None,
                injection_override=args.injections.resolve() if args.injections else None,
                stop_after_stage=args.stop_after_stage,
                stop_event=stop_event,
            )
        )
    finally:
        _restore_signal_handlers(installed)
    print(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "status": result.report.get("status"),
                "report": str(result.report_path),
                "snapshot_status": (result.report.get("snapshot") or {}).get("status_code"),
                "summary": result.report.get("summary"),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return result.exit_code


def _cmd_generate(args: argparse.Namespace) -> int:
    fixture = generate_fixture(args.out, seed=args.seed, force=args.force)
    print(
        json.dumps(
            {
                "samples": str(fixture.samples_path),
                "config": str(fixture.config_path),
                "injections": str(fixture.injections_path),
                "seed": fixture.seed,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    left = read_json(args.left)
    right = read_json(args.right)
    result = compare_reports(left, right)
    if args.out:
        atomic_write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["equal"] else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    summary = run_demo(args.root, seed=args.seed, force=args.force)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reports_dir": summary["reports_dir"],
                "parity": {
                    key: value["equal"] for key, value in summary["parity"].items()
                },
                "injection_mapping": summary["injection_mapping"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "demo":
            return _cmd_demo(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

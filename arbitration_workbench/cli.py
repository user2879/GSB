"""Command line interface for the arbitration workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .editing import apply_edit_commands, load_edit_commands
from .engine import run_workbench
from .sample import DEFAULT_SEED, generate_sample


def _write_json(path: Path | None, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    ledger = report.get("ledger", {})
    print(
        "fatal={fatal} workorders={workorders} ok={ok} failures={fail} pending={pending} "
        "boundary={boundary} budget_used={budget_used}".format(
            fatal=report.get("fatal"),
            workorders=summary.get("workorder_count", 0),
            ok=summary.get("successful_count", 0),
            fail=summary.get("failure_count", 0),
            pending=summary.get("pending_count", 0),
            boundary=summary.get("boundary_count", 0),
            budget_used=ledger.get("budget", {}).get("consumed_this_run", 0),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arbitration-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic synthetic sample")
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run = subparsers.add_parser("run", help="run a full or incremental arbitration batch")
    run.add_argument("--data", required=True, type=Path)
    run.add_argument("--state-dir", required=True, type=Path)
    run.add_argument("--report", type=Path)
    run.add_argument("--mode", choices=("serial", "parallel"), default="parallel")
    run.add_argument("--injections", type=Path)
    run.add_argument("--rebuild-state", action="store_true")
    run.add_argument("--quiet", action="store_true")

    edit = subparsers.add_parser("apply-edits", help="apply deterministic edit commands")
    edit.add_argument("--data", required=True, type=Path)
    edit.add_argument("--edits", required=True, type=Path)
    edit.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = generate_sample(args.out, seed=args.seed)
        _write_json(None, result)
        return 0
    if args.command == "run":
        report = run_workbench(
            args.data,
            args.state_dir,
            mode=args.mode,
            injections_path=args.injections,
            rebuild_state=args.rebuild_state,
        )
        _write_json(args.report, report)
        if not args.quiet:
            _print_summary(report)
        return 2 if report.get("fatal") else 0
    if args.command == "apply-edits":
        commands = load_edit_commands(args.edits)
        rows = apply_edit_commands(args.data, commands)
        _write_json(args.output, {"applied": rows, "count": len(rows)})
        return 0
    parser.error("unknown command")
    return 2

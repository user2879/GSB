"""Command line entry point for the hot-block diagnostic application."""
from __future__ import annotations

import argparse
import json
import os

from .config import (load_commands, load_fixedpoint, load_prefilter,
                     validate_candidates)
from .engine import DiagnosticEngine
from .errors import E_PACK_TRUNCATED, FailureList
from .injection import load_injections


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hotblock-diag",
        description="Long-sequence sparse hot-block fixed-point operator diagnostics",
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--pack", default="tensor_pack.bin")
    parser.add_argument("--prefilter", default="prefilter.json")
    parser.add_argument("--fixedpoint", default="fixedpoint.json")
    parser.add_argument("--commands", default="commands.json")
    parser.add_argument("--injections", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any persisted snapshot/cache and rebuild from blank")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    failures = FailureList()
    sample_dir = args.sample_dir
    pre = load_prefilter(os.path.join(sample_dir, args.prefilter), failures)
    fixed = load_fixedpoint(os.path.join(sample_dir, args.fixedpoint), failures)
    if pre is None or fixed is None:
        _emit_fatal(args.workspace, failures)
        return 2

    engine = DiagnosticEngine(
        workspace=args.workspace, pre=pre, fixed=fixed,
        injections=load_injections(args.injections, failures),
        failures=failures, fresh=args.fresh,
    )
    pack_path = os.path.join(sample_dir, args.pack)
    if not engine.load_pack(pack_path):
        _emit_fatal(args.workspace, failures)
        return 2
    if not engine.build_layout():
        _emit_fatal(args.workspace, failures)
        return 2
    validate_candidates(pre, engine.state.layout, failures)
    if any(item.code == "E_CONFIG_INVALID" for item in failures.failures):
        _emit_fatal(args.workspace, failures)
        return 2

    commands_path = os.path.join(sample_dir, args.commands)
    commands = load_commands(
        commands_path if os.path.exists(commands_path) else None,
        engine.state.layout, pre, fixed, failures,
    )
    if any(item.code == "E_COMMAND_REF" for item in failures.failures):
        _emit_fatal(args.workspace, failures)
        return 2
    engine.apply_edits(commands)
    if engine.state.layout is None:
        if not engine.load_pack(pack_path):
            _emit_fatal(args.workspace, failures)
            return 2
        if not engine.build_layout():
            _emit_fatal(args.workspace, failures)
            return 2
    validate_candidates(pre, engine.state.layout, failures)

    report = engine.run_to_completion(max_batches=args.max_batches)
    exit_code = 0 if report["status"] != "budget_paused" else 3
    if any(item.code == E_PACK_TRUNCATED for item in failures.failures):
        exit_code = 4
    print(json.dumps({
        "report": os.path.join(args.workspace, "diagnostic_report.json"),
        "status": report["status"],
        "rounds_completed": report["rounds_completed"],
        "failure_codes": failures.codes(),
        "exit_code": exit_code,
    }, sort_keys=True))
    return exit_code


def _emit_fatal(workspace: str, failures: FailureList) -> None:
    os.makedirs(workspace, exist_ok=True)
    path = os.path.join(workspace, "diagnostic_report.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"status": "fatal", "failures": failures.to_list()},
                  handle, sort_keys=True, indent=2)
        handle.write("\n")


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

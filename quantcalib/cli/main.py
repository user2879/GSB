"""quantcalib command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantcalib.data.config import load_config
from quantcalib.data.samples import generate_workspace
from quantcalib.engine import EngineContext, QuantCalibEngine
from quantcalib.report import write_report
from quantcalib.runner import (
    comparable_view,
    run_full_rebuild,
    run_incremental,
)


def _cmd_init(args: argparse.Namespace) -> None:
    paths = generate_workspace(Path(args.workdir))
    print(json.dumps(paths, indent=2, sort_keys=True))


def _cmd_run(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    cfg, issues = load_config(workdir / "config" / "quant.yaml")
    if issues:
        raise SystemExit(json.dumps([i.to_dict() for i in issues], indent=2))
    engine = QuantCalibEngine(EngineContext(workdir=workdir), cfg)
    result = engine.run(global_version=0)
    _emit(args, result)


def _cmd_stream(args: argparse.Namespace) -> None:
    results = run_incremental(args.workdir, args.commands)
    final = results[-1]
    rebuild = run_full_rebuild(args.workdir)
    equality = comparable_view(final) == comparable_view(rebuild)
    payload = {
        "incremental_final": final,
        "full_rebuild": rebuild,
        "field_level_equal": equality,
        "step_count": len(results),
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"field_level_equal": equality,
                      "steps": len(results)}, indent=2))


def _emit(args: argparse.Namespace, result: dict) -> None:
    if args.report:
        write_report(args.report, result)
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "global_version": result["global_version"],
        "resumed": result["resumed"],
        "layers": len(result["layers"]),
        "reuse": result["reuse"]["hit_count"],
        "recompute": result["reuse"]["recompute_count"],
        "failures": len(result["failures"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantcalib")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="generate deterministic samples")
    init.add_argument("--workdir", default=".")
    init.set_defaults(func=_cmd_init)

    run = sub.add_parser("run", help="single end-to-end run")
    run.add_argument("--workdir", default=".")
    run.add_argument("--out")
    run.add_argument("--report")
    run.set_defaults(func=_cmd_run)

    stream = sub.add_parser("stream", help="apply a command stream and verify")
    stream.add_argument("--workdir", default=".")
    stream.add_argument("--commands", required=True)
    stream.add_argument("--out")
    stream.set_defaults(func=_cmd_stream)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

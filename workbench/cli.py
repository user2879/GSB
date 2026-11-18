"""Command line entry point for the diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .errors import WorkbenchError
from .inject import INJECTIONS, prepare_injection
from .report import render_report
from .runner import Runner


def _print_error(error: WorkbenchError) -> int:
    payload = {"code": error.code, "message": error.message,
               "stage_id": error.stage_id, "field_path": error.field_path,
               "details": error.details}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")),
          file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workbench",
        description="Deterministic ML training pipeline diagnostic workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="validate and execute a pipeline")
    run_p.add_argument("spec", help="path to pipeline.json")
    run_p.add_argument("--workspace", required=True)
    run_p.add_argument("--report", help="write text report to this path")
    run_p.add_argument("--kill-after", type=int, default=None,
                       help="ordinal after which to simulate SIGKILL")

    rec_p = sub.add_parser("recover", help="replay committed stages and resume")
    rec_p.add_argument("spec")
    rec_p.add_argument("--workspace", required=True)
    rec_p.add_argument("--report")

    rep_p = sub.add_parser("replay", help="replay a run and verify the ledger")
    rep_p.add_argument("run_id", type=int)
    rep_p.add_argument("--workspace", required=True)

    inj_p = sub.add_parser("inject", help="inject a fault into inputs only")
    inj_p.add_argument("spec")
    inj_p.add_argument("--workspace", required=True)
    inj_p.add_argument("--kind", required=True, choices=INJECTIONS)
    inj_p.add_argument("--report")

    dump_p = sub.add_parser("ui-dump",
                            help="emit the headless UI view model as JSON")
    dump_p.add_argument("--workspace", required=True)
    dump_p.add_argument("--run-id", type=int)
    dump_p.add_argument("--out")

    sub.add_parser("ui", help="launch the Tkinter diagnostic window") \
        .add_argument("--workspace", required=True)

    sub.add_parser("check-env", help="verify python/numpy/tkinter/sqlite")
    return parser


def _write_report(runner: Runner, run_id: int, path: str | None) -> str:
    text = render_report(runner.ledger, run_id)
    if path:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-env":
        return _check_env()
    if args.command == "ui":
        from .ui import launch_ui
        launch_ui(args.workspace)
        return 0
    runner = Runner(args.workspace)
    try:
        if args.command == "run":
            result = runner.run_pipeline(args.spec, args.kill_after)
            if result["status"] == "COMPLETED":
                text = _write_report(runner, result["run_id"], args.report)
                print(text, end="")
            else:
                print(json.dumps(result, sort_keys=True))
        elif args.command == "recover":
            result = runner.recover(args.spec)
            text = _write_report(runner, result["new_run_id"], args.report)
            print(json.dumps(result, sort_keys=True))
            print(text, end="")
        elif args.command == "replay":
            result = runner.replay_run(args.run_id)
            print(json.dumps(result, sort_keys=True))
        elif args.command == "inject":
            injected_spec = prepare_injection(
                args.spec,
                os.path.join(args.workspace, "injections", args.kind),
                args.kind)
            result = runner.run_pipeline(injected_spec)
            text = _write_report(runner, result["run_id"], args.report)
            print(text, end="")
        elif args.command == "ui-dump":
            model = runner.ledger.view_model(args.run_id)
            text = json.dumps(model, indent=2, sort_keys=True)
            if args.out:
                with open(args.out, "w", encoding="utf-8",
                          newline="\n") as handle:
                    handle.write(text)
            print(text)
        return 0
    except WorkbenchError as error:
        return _print_error(error)
    finally:
        runner.close()


def _check_env() -> int:
    info = {"python": sys.version.split()[0], "sqlite": None,
            "numpy": None, "tkinter": None}
    import sqlite3
    info["sqlite"] = sqlite3.sqlite_version
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        info["numpy"] = "UNAVAILABLE"
    try:
        import tkinter
        info["tkinter"] = str(tkinter.TkVersion)
    except Exception:
        info["tkinter"] = "UNAVAILABLE"
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

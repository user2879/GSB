"""Command line entry points for replay, snapshots and environment checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .service import WorkbenchService
from .validation import load_master_records, load_replay_config


ROOT = Path(__file__).resolve().parents[1]


def _default_workspace(value: str | None) -> Path:
    return Path(value).resolve() if value else ROOT


def _default_writes(workspace: Path, value: str | None) -> Path:
    return Path(value).resolve() if value else workspace / "samples" / "writes.json"


def _default_config(workspace: Path, value: str | None) -> Path:
    return Path(value).resolve() if value else workspace / "samples" / "replay_params.json"


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _handle_error(service: WorkbenchService | None, exc: WorkbenchError) -> int:
    report = exc.as_dict()
    if service is not None:
        service.logger.event("business_error", **report)
    _print(report)
    if exc.code == "E_LOG":
        return 2
    if exc.code == "E_PARAM":
        return 3
    if exc.code == "E_SNAPSHOT":
        return 4
    if exc.code == "E_REPLAY":
        return 5
    return 1


def command_run(args: argparse.Namespace) -> int:
    workspace = _default_workspace(args.workspace)
    service: WorkbenchService | None = None
    try:
        records = load_master_records(_default_writes(workspace, args.writes))
        params, spec, _ = load_replay_config(_default_config(workspace, args.config))
        service = WorkbenchService(workspace)
        log = service.import_master_log(records, source=str(_default_writes(workspace, args.writes)))
        run = service.run_replay(
            params,
            spec,
            crash_after_completed_batches=args.crash_after,
        )
        estimation = service.estimate(params.version)
        result = {"log": log, "run": run, "estimation": estimation}
        _print(result)
        return 0
    except WorkbenchError as exc:
        return _handle_error(service, exc)
    finally:
        if service is not None:
            service.close()


def command_verify(args: argparse.Namespace) -> int:
    workspace = _default_workspace(args.workspace)
    service: WorkbenchService | None = None
    try:
        service = WorkbenchService(workspace)
        report = service.verify_snapshot(Path(args.snapshot).resolve())
        _print(report)
        return 0 if report["reproducible"] else 5
    except WorkbenchError as exc:
        return _handle_error(service, exc)
    finally:
        if service is not None:
            service.close()


def command_restore(args: argparse.Namespace) -> int:
    workspace = _default_workspace(args.workspace)
    service: WorkbenchService | None = None
    try:
        service = WorkbenchService(workspace)
        report = service.restore_snapshot(Path(args.snapshot).resolve())
        _print(report)
        return 0
    except WorkbenchError as exc:
        return _handle_error(service, exc)
    finally:
        if service is not None:
            service.close()


def command_state(args: argparse.Namespace) -> int:
    workspace = _default_workspace(args.workspace)
    service: WorkbenchService | None = None
    try:
        service = WorkbenchService(workspace)
        _print(service.state_summary())
        return 0
    except WorkbenchError as exc:
        return _handle_error(service, exc)
    finally:
        if service is not None:
            service.close()


def command_ui_check(args: argparse.Namespace) -> int:
    tk_spec = importlib.util.find_spec("tkinter")
    numpy_spec = importlib.util.find_spec("numpy")
    report = {
        "tkinter_available": tk_spec is not None,
        "numpy_available": numpy_spec is not None,
        "python": sys.version,
        "note": "Import-only check; opening a Tk window was not attempted.",
    }
    if tk_spec is not None and args.open_window:
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            root.destroy()
            report["window_created"] = True
        except Exception as exc:  # noqa: BLE001
            report["window_created"] = False
            report["window_error"] = str(exc)
    _print(report)
    return 0 if report["tkinter_available"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m workbench.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="import log, replay and estimate")
    run.add_argument("--workspace", default=None)
    run.add_argument("--writes", default=None)
    run.add_argument("--config", default=None)
    run.add_argument(
        "--crash-after",
        type=int,
        default=None,
        help="test-only: os._exit after this many committed batches and the next RUNNING marker",
    )
    run.set_defaults(func=command_run)

    verify = subparsers.add_parser("verify", help="replay a snapshot and compare every item")
    verify.add_argument("--workspace", default=None)
    verify.add_argument("--snapshot", required=True)
    verify.set_defaults(func=command_verify)

    restore = subparsers.add_parser("restore", help="whole-replace replay state from a snapshot")
    restore.add_argument("--workspace", default=None)
    restore.add_argument("--snapshot", required=True)
    restore.set_defaults(func=command_restore)

    state = subparsers.add_parser("state", help="print persisted state summary")
    state.add_argument("--workspace", default=None)
    state.set_defaults(func=command_state)

    ui_check = subparsers.add_parser("ui-check", help="check Tkinter/NumPy import availability")
    ui_check.add_argument("--open-window", action="store_true")
    ui_check.set_defaults(func=command_ui_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

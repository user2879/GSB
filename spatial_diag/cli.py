"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import json_safe
from .render import render_committed_snapshot
from .service import run_workbench
from .snapshot import SnapshotStore


DEFAULT_WORKSPACE = "artifacts"


def _print_json(value: Any) -> None:
    print(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, indent=2))


def _run(args: argparse.Namespace) -> int:
    report = run_workbench(
        args.workspace,
        args.config,
        render_view=not args.no_render,
    )
    _print_json(report)
    return 0


def _verify(args: argparse.Namespace) -> int:
    store = SnapshotStore(Path(args.workspace))
    store.recover()
    result = store.verify_current()
    _print_json(result)
    return 0 if result["ok"] else 2


def _render(args: argparse.Namespace) -> int:
    store = SnapshotStore(Path(args.workspace))
    store.recover()
    result = render_committed_snapshot(store)
    _print_json(
        {
            "version": result["version"],
            "view_dir": str(result["view_dir"]),
            "files": result["files"],
        }
    )
    return 0


def _qt(args: argparse.Namespace) -> int:
    from .qt_view import launch_viewer

    return launch_viewer(args.workspace, offscreen=args.offscreen)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial-diag",
        description="Deterministic spatial acoustic reconstruction diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="generate, reconstruct, validate, snapshot, render")
    run.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    run.add_argument("--config", default=None)
    run.add_argument("--no-render", action="store_true")
    run.set_defaults(handler=_run)

    verify = subparsers.add_parser("verify", help="verify snapshot and artifact content hashes")
    verify.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    verify.set_defaults(handler=_verify)

    render = subparsers.add_parser("render", help="render the current immutable snapshot")
    render.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    render.set_defaults(handler=_render)

    qt = subparsers.add_parser("qt", help="open the PyQt5 diagnostic view")
    qt.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    qt.add_argument("--offscreen", action="store_true")
    qt.set_defaults(handler=_qt)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["run"]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())


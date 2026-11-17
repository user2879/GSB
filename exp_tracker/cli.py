"""Command-line interface with stable JSON reports and business error codes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .demo import run_demo
from .errors import FormatError, StateError, TrackerError
from .repository import UpgradeInterrupted
from .service import ExperimentService
from .util import atomic_write_json


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)


def _emit(value: Any) -> None:
    print(_json(value))


def _workspace(args: argparse.Namespace) -> Path:
    return Path(args.workspace).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exptrack",
        description="Deterministic local experiment records, upgrades and snapshots",
    )
    parser.add_argument(
        "--workspace",
        default="demo_workspace",
        help="workspace directory (default: demo_workspace)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_workspace_option(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--workspace",
            default=argparse.SUPPRESS,
            help="workspace directory (also accepted before the command)",
        )

    init = subparsers.add_parser("init", help="initialize a session from local JSON")
    add_workspace_option(init)
    init.add_argument("--from", dest="source", required=True, help="declaration JSON path")
    init.add_argument("--force", action="store_true", help="replace the app-owned session state")

    add_workspace_option(
        subparsers.add_parser("status", help="show the complete workspace status")
    )
    add_workspace_option(subparsers.add_parser("list", help="list current runs"))
    add_workspace_option(subparsers.add_parser("snapshots", help="list saved snapshots"))

    show = subparsers.add_parser("show", help="show one run and its history")
    add_workspace_option(show)
    show.add_argument("run_id")

    curve = subparsers.add_parser("curve", help="show stable ASCII metric curves")
    add_workspace_option(curve)
    curve.add_argument("run_id")

    upgrade = subparsers.add_parser("upgrade", help="open the library and continue upgrades")
    add_workspace_option(upgrade)
    upgrade.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="test hook: stop after N newly upgraded records",
    )

    commit = subparsers.add_parser("commit", help="submit a full v3 replacement record")
    add_workspace_option(commit)
    commit.add_argument("--record", required=True, help="record JSON file")

    snapshot = subparsers.add_parser("snapshot", help="save the whole session state")
    add_workspace_option(snapshot)
    snapshot.add_argument("--label", required=True)

    restore = subparsers.add_parser("restore", help="replace state from a snapshot")
    add_workspace_option(restore)
    restore.add_argument("--id", type=int, required=True, dest="snapshot_id")
    restore.add_argument("--no-replay", action="store_true")

    replay = subparsers.add_parser("replay", help="replay and verify one snapshot")
    add_workspace_option(replay)
    replay.add_argument("--id", type=int, required=True, dest="snapshot_id")

    demo = subparsers.add_parser("demo", help="run the three deterministic scenarios")
    add_workspace_option(demo)
    demo.add_argument("--output", help="also write the demo report JSON here")

    add_workspace_option(
        subparsers.add_parser("gui", help="launch the Tkinter workbench")
    )
    return parser


def _load_record(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StateError("record file does not exist", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise FormatError(
            "$", f"invalid JSON: {exc.msg}", line=exc.lineno, column=exc.colno, source=str(path)
        ) from exc
    if not isinstance(raw, dict):
        raise FormatError("$", "record document must be a JSON object")
    return raw


def _ascii_curve(
    points: list[dict[str, Any]], field: str, *, width: int = 48, height: int = 9
) -> list[str]:
    if not points:
        return ["<no metric points>"]
    values = [float(point[field]) for point in points]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        maximum = minimum + 1.0
    span = maximum - minimum
    columns = max(2, width)
    rows: list[list[str]] = [[" " for _ in range(columns)] for _ in range(height)]
    for index, value in enumerate(values):
        column = int(round(index * (columns - 1) / max(len(values) - 1, 1)))
        row = int(round((maximum - value) * (height - 1) / span))
        rows[row][column] = "#"
    lines = [f"{maximum:9.6f} |" + "".join(rows[0])]
    lines.extend("          |" + "".join(row) for row in rows[1:-1])
    lines.append(f"{minimum:9.6f} |" + "".join(rows[-1]))
    lines.append("          +" + "-" * columns)
    lines.append(
        f"           step 1{' ' * max(1, columns - 12)}step {points[-1]['step']}"
    )
    return lines


def _run_command(args: argparse.Namespace) -> Any:
    workspace = _workspace(args)
    if args.command == "init":
        service = ExperimentService.initialize(
            workspace, Path(args.source), force=args.force
        )
        return {
            "ok": True,
            "status": "initialized",
            "workspace": str(workspace),
            "library_id": service.library.library_id,
            "upgrade": service.upgrade_result.to_dict(),
            "runs": service.list_runs(),
        }
    if args.command == "demo":
        result = run_demo(workspace)
        if args.output:
            atomic_write_json(Path(args.output).resolve(), result)
        return {"ok": True, **result}
    if args.command == "gui":
        from .gui import launch

        launch(workspace)
        return {"ok": True, "status": "gui_closed", "workspace": str(workspace)}

    service = ExperimentService.open(workspace)
    if args.command == "status":
        return {"ok": True, **service.status()}
    if args.command == "list":
        return {
            "ok": True,
            "workspace": str(workspace),
            "runs": service.list_runs(),
            "format_versions": service.library.format_versions(),
            "source_format_versions": service.library.source_format_versions(),
            "aggregation_versions": service.library.aggregation_versions(),
        }
    if args.command == "show":
        detail = service.run_detail(args.run_id)
        return {
            "ok": True,
            "run": detail,
            "charts": {
                "loss": _ascii_curve(detail["step_metrics"], "loss"),
                "accuracy": _ascii_curve(detail["step_metrics"], "accuracy"),
            },
        }
    if args.command == "curve":
        detail = service.run_detail(args.run_id)
        return {
            "ok": True,
            "run_id": args.run_id,
            "run_version": detail["run"]["run_version"],
            "charts": {
                "loss": _ascii_curve(detail["step_metrics"], "loss"),
                "accuracy": _ascii_curve(detail["step_metrics"], "accuracy"),
            },
        }
    if args.command == "upgrade":
        return {"ok": True, "upgrade": service.upgrade_result.to_dict()}
    if args.command == "commit":
        return {"ok": True, "commit": service.commit_record(_load_record(args.record))}
    if args.command == "snapshot":
        return {"ok": True, "snapshot": service.snapshot(args.label)}
    if args.command == "snapshots":
        return {"ok": True, "snapshots": service.list_snapshots()}
    if args.command == "restore":
        return {
            "ok": True,
            "restore": service.restore_snapshot(
                args.snapshot_id, replay=not args.no_replay
            ),
        }
    if args.command == "replay":
        return {"ok": True, "replay": service.replay_snapshot(args.snapshot_id)}
    raise StateError("unknown command", command=args.command)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _emit(_run_command(args))
        return 0
    except UpgradeInterrupted as exc:
        _emit(
            {
                "ok": False,
                "status": "interrupted",
                "checkpoint": str(exc.checkpoint_path),
                "last_completed": exc.entry,
            }
        )
        return 3
    except TrackerError as exc:
        _emit(exc.to_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .commands import validate_command_stream
from .configuration import load_config
from .demo import _environment_report, run_demo
from .errors import WorkbenchError
from .samples import generate_workspace
from .session import WorkbenchSession
from .snapshot import SnapshotStore
from .workflows import load_bundle, open_snapshot_session


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_default))


def _default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workbench", description="Layered perturbation budget diagnostic workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-samples", help="generate deterministic PNG, masks and JSON examples")
    sub.add_parser("check-env", help="report Python, NumPy, Pillow and tkinter availability")
    sub.add_parser("demo", help="run the full local acceptance demonstration")
    sub.add_parser("gui", help="open the optional tkinter diagnostic workbench")

    validate = sub.add_parser("validate", help="validate config, PNG, mask and optional command stream")
    validate.add_argument("--config", default="configs/workbench.json")
    validate.add_argument("--commands", default=None)

    snapshot = sub.add_parser("snapshot-check", help="check committed snapshot version/config hash")
    snapshot.add_argument("--config", default="configs/workbench.json")
    snapshot.add_argument("--state-dir", default=None)

    replay = sub.add_parser("run-stream", help="validate and replay the command stream")
    replay.add_argument("--config", default="configs/workbench.json")
    replay.add_argument("--commands", default="configs/edit_stream.json")
    replay.add_argument("--state-dir", default=None)
    replay.add_argument("--budget", type=int, default=None)
    replay.add_argument("--workers", type=int, default=None)
    replay.add_argument("--clean", action="store_true", help="clear snapshot state before replay")
    replay.add_argument("--recover-stale", action="store_true", help="discard stale pointer and perform whole-snapshot recovery")
    replay.add_argument("--drain", action="store_true", help="continue batches until pending queue is empty")
    replay.add_argument("--max-batches-per-command", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate-samples":
            paths = generate_workspace(Path.cwd())
            print_json({"ok": True, "generated": len(paths), "paths": [str(p) for p in paths]})
            return 0
        if args.command == "check-env":
            print_json({"ok": True, "environment": _environment_report()})
            return 0
        if args.command == "demo":
            report = run_demo(Path.cwd())
            summary = {
                "ok": report["ok"],
                "report": "artifacts/reports/verification-report.json",
                "consistency_equal": report["consistency"]["equal"],
                "injection_consistent": report["injection_diagnostic"]["consistent"],
                "all_injections_hit": report["injection_diagnostic"]["all_injections_hit"],
                "final": report["outputs"]["incremental_final"],
            }
            print_json(summary)
            return 0 if report["ok"] else 4
        if args.command == "gui":
            from .gui import launch
            return launch(Path.cwd())
        if args.command == "validate":
            config, rectangles, pixels, fingerprint = load_bundle(args.config)
            command_count = 0
            if args.commands:
                commands = validate_command_stream(args.commands, config, rectangles)
                command_count = len(commands)
            print_json({
                "ok": True,
                "config_hash": config.config_hash,
                "global_version": config.global_version,
                "image_fingerprint": fingerprint,
                "mask_rectangles": len(rectangles),
                "levels": config.levels,
                "tile_count": sum(config.tile_grid(level)[0] * config.tile_grid(level)[1]
                                  for level in range(config.levels)),
                "commands": command_count,
            })
            return 0
        if args.command == "snapshot-check":
            config = load_config(args.config)
            store = SnapshotStore(config, args.state_dir)
            manifest = store.load_manifest()
            store.check_compatible(manifest)
            if manifest is None:
                print_json({"ok": True, "snapshot": None, "status": "blank"})
            else:
                print_json({
                    "ok": True,
                    "status": "compatible",
                    "global_version": manifest["global_version"],
                    "config_hash": manifest["config_hash"],
                    "command_cursor": manifest.get("command_cursor"),
                    "applied_command_cursor": manifest.get("applied_command_cursor"),
                    "pending": len(manifest.get("pending", [])),
                    "snapshot_dir": manifest.get("snapshot_dir"),
                })
            return 0
        if args.command == "run-stream":
            return _run_stream(args)
        raise WorkbenchError("E_CONFIG_INVALID", f"unknown command {args.command}")
    except WorkbenchError as exc:
        print_json(exc.to_dict())
        return exc.exit_code


def _run_stream(args: argparse.Namespace) -> int:
    config, rectangles, pixels, fingerprint = load_bundle(args.config)
    commands = validate_command_stream(args.commands, config, rectangles)
    if args.budget is not None and (not isinstance(args.budget, int) or args.budget <= 0):
        raise WorkbenchError("E_CONFIG_INVALID", "--budget must be a positive integer")
    if args.workers is not None and not (1 <= args.workers <= config.max_workers):
        raise WorkbenchError("E_CONFIG_INVALID", "--workers must be between 1 and configured max_workers")

    store = SnapshotStore(config, args.state_dir)
    if args.clean:
        store.reset_for_demo(remove_all=True)
    try:
        session, _ = open_snapshot_session(config, pixels, fingerprint, rectangles, store)
    except WorkbenchError as exc:
        if exc.code != "E_SNAPSHOT_STALE" or not args.recover_stale:
            raise
        store.reset_for_demo(remove_all=True)
        session = WorkbenchSession.fresh(config, pixels, fingerprint, rectangles)
        stale_recovered = True
    else:
        stale_recovered = False
    start_index = session.applied_cursor
    budget = args.budget or config.tile_budget
    workers = args.workers or config.max_workers

    batch_reports = []
    for command_index in range(start_index, len(commands)):
        command = commands[command_index]
        session.apply(command)
        for _batch in range(max(1, args.max_batches_per_command)):
            report = session.run_batch(budget=budget, max_workers=workers)
            manifest = session.commit(store)
            batch_reports.append({
                "command_index": command_index,
                "command_id": command.get("id"),
                "status": report["status"],
                "reused": len(report["reused"]),
                "recomputed": len(report["recomputed"]),
                "pending": len(report["pending"]),
                "snapshot": manifest["snapshot_dir"],
            })
            if not args.drain or not report["pending"]:
                break
    final_status = batch_reports[-1]["status"] if batch_reports else "no_commands"
    print_json({
        "ok": final_status == "complete" or final_status == "no_commands",
        "final_status": final_status,
        "applied_cursor": session.applied_cursor,
        "completed_cursor": session.completed_cursor,
        "pending": session.pending,
        "stale_recovered": stale_recovered,
        "batches": batch_reports,
        "manifest": str(store.manifest_path),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())

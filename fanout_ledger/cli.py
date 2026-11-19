"""Command line entry point for the fan-out ledger component.

Subcommands:
    validate   strict-load a scene file
    run        start mock services, run a fan-out session, write ledger+snapshot
    replay     load a snapshot and verify deterministic replay
    ui         launch the Tkinter dashboard (optional; headless-safe import)
    init-db    create the SQLite file and snapshot directory and check writability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_scene
from .engine import new_session_id, render_report, run_session
from .errors import LedgerError
from .ledger import Ledger
from .logging_util import StructuredLogger
from .mock_service import start_all, stop_all
from .recovery import verify_replay
from .snapshot import list_snapshots, load_snapshot, verify_compatibility


def _logger(args: argparse.Namespace) -> StructuredLogger:
    log_dir = Path(args.run_dir) if getattr(args, "run_dir", None) else Path("run")
    return StructuredLogger(log_dir / "events.jsonl")


def cmd_validate(args: argparse.Namespace) -> int:
    logger = StructuredLogger(Path("run") / "events.jsonl")
    try:
        scene = load_scene(args.scene)
    except LedgerError as exc:
        logger.error("scene_rejected", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 3
    logger.info("scene_accepted", scene=args.scene,
                scene_version=scene.scene_version, nodes=len(scene.nodes))
    print(json.dumps({"code": "OK", "scene_version": scene.scene_version,
                      "nodes": [n.node_id for n in scene.nodes],
                      "payload_digest": scene.payload_digest}, indent=2))
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    snapshot_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    probe = run_dir / ".write-probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    session_id = new_session_id("init")
    ledger = Ledger(run_dir / "ledger.sqlite3", session_id)
    meta = ledger.fetch_meta()
    ledger.close()
    print(json.dumps({"code": "OK", "db": str(run_dir / "ledger.sqlite3"),
                      "snapshot_dir": str(snapshot_dir), "meta": meta}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logger = _logger(args)
    try:
        scene = load_scene(args.scene)
    except LedgerError as exc:
        logger.error("scene_rejected", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 3
    handles = start_all(scene.nodes)
    try:
        result = run_session(
            scene, run_dir=args.run_dir, session_id=args.session_id,
            snapshot_every=args.snapshot_every, logger=logger,
        )
    finally:
        stop_all(handles)
    print(render_report(result, scene))
    (Path(args.run_dir) / "report.txt").write_text(
        render_report(result, scene), encoding="utf-8")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    logger = _logger(args)
    scene = load_scene(args.scene)
    snapshot_path = Path(args.snapshot) if args.snapshot else \
        sorted(list_snapshots(Path(args.run_dir) / "snapshots"))[-1]
    try:
        snapshot = load_snapshot(snapshot_path)
        verify_compatibility(
            snapshot, scene.scene_version, scene.payload_digest,
            [
                {
                    "node_id": n.node_id, "failure_rate": n.failure_rate,
                    "slow_rate": n.slow_rate, "process_ms": n.process_ms,
                    "slow_extra_ms": n.slow_extra_ms, "seed": n.seed,
                }
                for n in scene.nodes
            ],
        )
    except LedgerError as exc:
        logger.error("replay_blocked", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 3
    handles = start_all(scene.nodes)
    try:
        outcome = verify_replay(scene, snapshot)
    except LedgerError as exc:
        stop_all(handles)
        logger.error("replay_diverged", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 4
    finally:
        if handles:
            stop_all(handles)
    logger.info("replay_verified", snapshot=str(snapshot_path), **outcome)
    print(json.dumps({"code": "OK", "snapshot": str(snapshot_path),
                      "replay": outcome}, indent=2))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .ui.dashboard import launch_dashboard
        launch_dashboard(args.run_dir)
    except ImportError as exc:
        payload = {"code": "E_UI_UNAVAILABLE",
                   "message": "Tkinter is not available in this runtime",
                   "detail": str(exc)}
        log_dir = Path(args.run_dir) if Path(args.run_dir).exists() else Path("run")
        log_dir.mkdir(parents=True, exist_ok=True)
        StructuredLogger(log_dir / "events.jsonl", echo=False).error(
            "ui_unavailable", **payload)
        print(json.dumps(payload, indent=2))
        return 5
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fanout-ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="strictly validate a scene file")
    p_validate.add_argument("scene")
    p_validate.set_defaults(func=cmd_validate)

    p_init = sub.add_parser("init-db", help="initialize SQLite ledger and snapshot dir")
    p_init.add_argument("run_dir")
    p_init.set_defaults(func=cmd_init_db)

    p_run = sub.add_parser("run", help="run a fan-out session")
    p_run.add_argument("scene")
    p_run.add_argument("--run-dir", default="run/session")
    p_run.add_argument("--session-id", default=None)
    p_run.add_argument("--snapshot-every", type=int, default=0)
    p_run.set_defaults(func=cmd_run)

    p_replay = sub.add_parser("replay", help="verify deterministic replay of a snapshot")
    p_replay.add_argument("scene")
    p_replay.add_argument("--run-dir", default="run/session")
    p_replay.add_argument("--snapshot", default=None)
    p_replay.set_defaults(func=cmd_replay)

    p_ui = sub.add_parser("ui", help="launch the Tkinter dashboard")
    p_ui.add_argument("--run-dir", default="run/session")
    p_ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LedgerError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

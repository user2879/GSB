"""Command-line entry points for validation, execution, snapshots, and replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import Engine, EngineOptions
from .errors import WorkbenchError
from .model import SessionSpec, load_session
from .replay import replay_from_snapshot
from .snapshot import SnapshotStore


DEFAULT_DB = "snapshots.sqlite3"


def _print_json(value: Any, *, stream: Any = sys.stdout) -> None:
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")


def _advance_to_event(engine: Engine, event_seq: int) -> None:
    while engine.event_cursor < len(engine.spec.events):
        event = engine.spec.events[engine.event_cursor]
        if event.seq > event_seq:
            break
        engine.step()
    if engine.event_cursor == 0:
        raise WorkbenchError(
            "E_SPEC",
            f"event sequence {event_seq} precedes the first event",
            field="events",
            event_seq=event_seq,
        )


def _engine_snapshot(engine: Engine, snapshot_db: str, label: str, *, read_only: bool = False) -> dict[str, Any]:
    with SnapshotStore(snapshot_db) as store:
        if read_only:
            info = store._insert(engine, label, read_only=True)
        else:
            info = store.save(engine, label)
        return info.to_dict()


def _summary_with_details(engine: Engine, full: bool) -> dict[str, Any]:
    summary = engine.summary()
    if full:
        summary.update(
            {
                "frame_trace": [frame.to_dict() for frame in engine.frames],
                "transitions": [item.to_dict() for item in engine.transitions],
                "fairness": engine.fairness_report(),
                "starvation_warnings": engine.starvation_warnings(),
                "strategy_switches": [item.to_dict() for item in engine.strategy_switches],
                "final_streams": [engine.streams[stream_id].to_dict() for stream_id in sorted(engine.streams)],
            }
        )
    return summary


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_session(args.session)
    _print_json(
        {
            "ok": True,
            "session_id": spec.session_id,
            "declared_version": spec.declared_version,
            "content_digest": spec.content_digest,
            "streams": len(spec.streams),
            "events": len(spec.events),
            "event_offset": spec.event_offset,
            "next_event_seq": spec.next_event_seq,
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_session(args.session)
    engine = Engine(spec, options=EngineOptions())
    if args.snapshot_at is not None:
        _advance_to_event(engine, args.snapshot_at)
        snapshot = _engine_snapshot(engine, args.db, args.label or f"event-{engine.next_event_seq - 1}")
        engine.run()
    else:
        engine.run()
        snapshot = None
        if args.snapshot:
            snapshot = _engine_snapshot(engine, args.db, args.label or "final")
    value = _summary_with_details(engine, args.full)
    if snapshot is not None:
        value["snapshot"] = snapshot
    if args.output:
        Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json(value)
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    spec = load_session(args.session)
    engine = Engine(spec)
    if args.at is not None:
        _advance_to_event(engine, args.at)
    else:
        engine.run()
    snapshot = _engine_snapshot(engine, args.db, args.label)
    _print_json({"ok": True, "snapshot": snapshot, "summary": engine.summary()})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    spec = load_session(args.session)
    reference = Engine(spec).run()
    with SnapshotStore(args.db) as store:
        result = replay_from_snapshot(
            store,
            args.snapshot,
            reference_spec=spec,
            reference_engine=reference,
        )
    _print_json({"ok": True, "replay": result.to_dict()})
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    spec = load_session(args.session)
    current = Engine(spec).run()
    with SnapshotStore(args.db) as store:
        result = store.restore(args.snapshot, spec, current)
    _print_json({"ok": True, "restore": result.to_dict(), "summary": result.engine.summary()})
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        raise WorkbenchError(
            "E_IO",
            f"Tkinter is unavailable: {exc}. Use the CLI commands on this machine.",
            field="tkinter",
        ) from exc
    from .gui import launch_gui

    try:
        launch_gui(args.session, snapshot_db=args.db)
    except Exception as exc:
        raise WorkbenchError(
            "E_IO",
            f"Tkinter workbench could not start: {exc}",
            field="tkinter",
        ) from exc
    return 0


def cmd_list_snapshots(args: argparse.Namespace) -> int:
    with SnapshotStore(args.db) as store:
        snapshots = store.list_snapshots()
    _print_json({"ok": True, "snapshots": snapshots})
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from tools.verify import main as verify_main

    argv = ["--output", args.output]
    if args.skip_tests:
        argv.append("--skip-tests")
    return verify_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="h2burst",
        description="Local HTTP/2 burst-window diagnostic workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a session declaration")
    validate.add_argument("session")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run a session deterministically")
    run.add_argument("session")
    run.add_argument("--db", default=DEFAULT_DB)
    run.add_argument("--snapshot", action="store_true", help="save a final snapshot")
    run.add_argument("--snapshot-at", type=int, help="save a snapshot after this event sequence")
    run.add_argument("--label", help="snapshot label")
    run.add_argument("--output", help="also write JSON output to this path")
    run.add_argument("--full", action="store_true", help="include complete frame and transition traces")
    run.set_defaults(func=cmd_run)

    snapshot = sub.add_parser("snapshot", help="save a snapshot")
    snapshot.add_argument("session")
    snapshot.add_argument("--db", default=DEFAULT_DB)
    snapshot.add_argument("--at", type=int, help="event sequence to stop after")
    snapshot.add_argument("--label", default="manual")
    snapshot.set_defaults(func=cmd_snapshot)

    replay = sub.add_parser("replay", help="replay a snapshot frame by frame")
    replay.add_argument("session")
    replay.add_argument("snapshot", type=int)
    replay.add_argument("--db", default=DEFAULT_DB)
    replay.set_defaults(func=cmd_replay)

    restore = sub.add_parser("restore", help="restore or wholly replace from a snapshot")
    restore.add_argument("session")
    restore.add_argument("snapshot", type=int)
    restore.add_argument("--db", default=DEFAULT_DB)
    restore.set_defaults(func=cmd_restore)

    snapshots = sub.add_parser("snapshots", help="list snapshots")
    snapshots.add_argument("--db", default=DEFAULT_DB)
    snapshots.set_defaults(func=cmd_list_snapshots)

    gui = sub.add_parser("gui", help="launch the Tkinter workbench")
    gui.add_argument("session", nargs="?", default="samples/minimal.json")
    gui.add_argument("--db", default=DEFAULT_DB)
    gui.set_defaults(func=cmd_gui)

    verify = sub.add_parser("verify", help="run tests and write the verification artifact")
    verify.add_argument("--output", default="artifacts/verification.json")
    verify.add_argument("--skip-tests", action="store_true")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        _print_json({"ok": False, "error": exc.to_dict()}, stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

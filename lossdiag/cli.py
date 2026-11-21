"""Command-line interface for the loss diagnosis workbench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .echo import recover_echo_state, run_echo_replay, serve_peer
from .errors import DiagnosticError
from .events import load_event_bundle
from .pipeline import build_session
from .replay import verify_ledger
from .store import WorkbenchStore, restore_snapshot
from .util import atomic_write_text, content_digest, pretty_json
from .wire import read_wire_bundle


DEFAULT_DB = "artifacts/workbench.db"
DEFAULT_ECHO_DB = "artifacts/echo_state.db"


def _emit(value: Any, *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = pretty_json(value)
    sys.stdout.write(text + "\n")


def _maybe_write(path: str | None, value: Any) -> None:
    if path:
        atomic_write_text(path, pretty_json(value) + "\n")


def _add_source_args(parser: argparse.ArgumentParser, *, events_required: bool = True) -> None:
    parser.add_argument("--wire", required=True, help="compact packet observation file")
    parser.add_argument("--manifest", required=True, help="wire basis/manifest JSON")
    parser.add_argument(
        "--events",
        nargs="+",
        required=events_required,
        help="one or more loss-event list JSON files",
    )
    parser.add_argument("--window-ms", type=int, default=100, help="integer aggregation window")


def cmd_inspect_wire(args: argparse.Namespace) -> dict[str, Any]:
    bundle = read_wire_bundle(
        args.wire,
        args.manifest,
        corrupt=args.inject == "wire-corrupt",
    )
    return {
        "ok": True,
        "frame_count": len(bundle.frames),
        "source_id": bundle.source_id,
        "model_version": bundle.model_version,
        "time_basis": bundle.time_basis.to_dict(),
        "alignment_version": bundle.alignment_version,
        "raw_digest": bundle.raw_digest,
        "wire_digest": bundle.wire_digest,
        "frames": [frame.summary() for frame in bundle.frames],
    }


def cmd_inspect_events(args: argparse.Namespace) -> dict[str, Any]:
    bundles = [
        load_event_bundle(
            path,
            inject_version_conflict=args.inject == "event-version-conflict" and index == 0,
        )
        for index, path in enumerate(args.events)
    ]
    return {
        "ok": True,
        "event_file_count": len(bundles),
        "versions": sorted({bundle.version for bundle in bundles}),
        "bundles": [bundle.to_dict() for bundle in bundles],
    }


def _build_from_args(args: argparse.Namespace):
    return build_session(
        wire_path=args.wire,
        wire_manifest_path=args.manifest,
        event_paths=list(args.events),
        window_ms=args.window_ms,
        inject_wire_corruption=args.inject == "wire-corrupt",
        inject_event_version_conflict=args.inject == "event-version-conflict",
    )


def cmd_diagnose(args: argparse.Namespace) -> dict[str, Any]:
    session = _build_from_args(args)
    output: dict[str, Any] = dict(session.report)
    output["saved"] = {}
    if args.db or args.snapshot or args.ledger:
        db_path = args.db or DEFAULT_DB
        with WorkbenchStore(db_path) as store:
            if args.snapshot:
                output["saved"]["snapshot_uid"] = store.save_snapshot(session)
            if args.ledger:
                output["saved"]["ledger_uid"] = store.save_ledger(session)
        output["saved"]["db"] = db_path
    _maybe_write(args.out, output)
    return output


def cmd_snapshot_save(args: argparse.Namespace) -> dict[str, Any]:
    session = _build_from_args(args)
    with WorkbenchStore(args.db) as store:
        uid = store.save_snapshot(session)
    return {
        "ok": True,
        "snapshot_uid": uid,
        "content_digest": content_digest(session.model_payload()),
        "report_digest": session.report_digest,
        "report": session.report,
    }


def cmd_snapshot_list(args: argparse.Namespace) -> dict[str, Any]:
    with WorkbenchStore(args.db) as store:
        snapshots = store.list_snapshots()
    return {"ok": True, "snapshots": snapshots}


def cmd_snapshot_restore(args: argparse.Namespace) -> dict[str, Any]:
    current_session = None
    if args.wire or args.manifest or args.events:
        if not (args.wire and args.manifest and args.events):
            raise DiagnosticError(
                "E_FORMAT",
                "current session comparison requires wire, manifest and events together",
                record_number=0,
                field="current_session",
            )
        current_session = build_session(
            wire_path=args.wire,
            wire_manifest_path=args.manifest,
            event_paths=list(args.events),
            window_ms=args.window_ms,
        )
    with WorkbenchStore(args.db) as store:
        result = restore_snapshot(
            store,
            args.snapshot_id,
            current_session=current_session,
            inject_digest_mismatch=args.inject == "snapshot-digest-mismatch",
        )
        if args.verify_ledger_id:
            result["ledger_replay"] = verify_ledger(store, args.verify_ledger_id)
    _maybe_write(args.out, result)
    return result


def cmd_ledger_save(args: argparse.Namespace) -> dict[str, Any]:
    session = _build_from_args(args)
    with WorkbenchStore(args.db) as store:
        uid = store.save_ledger(session)
    return {
        "ok": True,
        "ledger_uid": uid,
        "input_digest": session.input_digest,
        "report_digest": session.report_digest,
    }


def cmd_ledger_list(args: argparse.Namespace) -> dict[str, Any]:
    with WorkbenchStore(args.db) as store:
        records = store.list_ledger()
    return {"ok": True, "ledger": records}


def cmd_ledger_verify(args: argparse.Namespace) -> dict[str, Any]:
    with WorkbenchStore(args.db) as store:
        result = verify_ledger(
            store,
            args.ledger_id,
            wire_path=args.wire,
            wire_manifest_path=args.manifest,
            event_paths=args.events,
            window_ms=args.window_ms,
        )
    _maybe_write(args.out, result)
    return result


def cmd_echo_serve(args: argparse.Namespace) -> dict[str, Any]:
    serve_peer(
        host="127.0.0.1",
        port=args.port,
        state_db=args.state_db,
        once=args.once,
        ack_timeout_seconds=args.ack_timeout,
    )
    return {"ok": True, "message": "echo peer stopped"}


def cmd_echo_replay(args: argparse.Namespace) -> dict[str, Any]:
    session = _build_from_args(args)
    request_id = args.request_id or f"echo-{session.report_digest[:16]}"
    result = run_echo_replay(
        session,
        host="127.0.0.1",
        port=args.port,
        request_id=request_id,
        send_ack=not args.no_ack,
        inject_mismatch=args.inject == "echo-mismatch",
    )
    _maybe_write(args.out, result)
    return result


def cmd_echo_recover(args: argparse.Namespace) -> dict[str, Any]:
    result = recover_echo_state(args.state_db)
    _maybe_write(args.out, result)
    return result


def cmd_generate_samples(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.generate_samples import main as generate_main

    return {"ok": True, "exit_code": generate_main()}


def cmd_gui(args: argparse.Namespace) -> dict[str, Any]:
    from .gui import main as gui_main

    return {"ok": True, "exit_code": gui_main(db_path=args.db)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lossdiag", description="Loss observation reconciliation workbench")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    wire = subparsers.add_parser("inspect-wire", help="decode and validate every observation frame")
    wire.add_argument("--wire", required=True)
    wire.add_argument("--manifest", required=True)
    wire.add_argument(
        "--inject",
        choices=("none", "wire-corrupt"),
        default="none",
    )
    wire.set_defaults(func=cmd_inspect_wire)

    events = subparsers.add_parser("inspect-events", help="normalize v1/v2/v3 event lists")
    events.add_argument("--events", nargs="+", required=True)
    events.add_argument(
        "--inject",
        choices=("none", "event-version-conflict"),
        default="none",
    )
    events.set_defaults(func=cmd_inspect_events)

    diagnose = subparsers.add_parser("diagnose", help="run the complete diagnosis pipeline")
    _add_source_args(diagnose)
    diagnose.add_argument("--db", default=None)
    diagnose.add_argument("--snapshot", action="store_true")
    diagnose.add_argument("--ledger", action="store_true")
    diagnose.add_argument("--out")
    diagnose.add_argument(
        "--inject",
        choices=("none", "wire-corrupt", "event-version-conflict"),
        default="none",
    )
    diagnose.set_defaults(func=cmd_diagnose)

    snapshot_save = subparsers.add_parser("snapshot-save", help="persist a diagnosis session snapshot")
    _add_source_args(snapshot_save)
    snapshot_save.add_argument("--db", default=DEFAULT_DB)
    snapshot_save.add_argument(
        "--inject",
        choices=("none", "wire-corrupt", "event-version-conflict"),
        default="none",
    )
    snapshot_save.set_defaults(func=cmd_snapshot_save)

    snapshot_list = subparsers.add_parser("snapshot-list", help="list persisted snapshots")
    snapshot_list.add_argument("--db", default=DEFAULT_DB)
    snapshot_list.set_defaults(func=cmd_snapshot_list)

    snapshot_restore = subparsers.add_parser("snapshot-restore", help="restore by wholesale replacement")
    snapshot_restore.add_argument("--db", default=DEFAULT_DB)
    snapshot_restore.add_argument("--snapshot-id", required=True)
    snapshot_restore.add_argument("--wire")
    snapshot_restore.add_argument("--manifest")
    snapshot_restore.add_argument("--events", nargs="+")
    snapshot_restore.add_argument("--window-ms", type=int, default=100)
    snapshot_restore.add_argument("--verify-ledger-id")
    snapshot_restore.add_argument("--out")
    snapshot_restore.add_argument(
        "--inject",
        choices=("none", "snapshot-digest-mismatch"),
        default="none",
    )
    snapshot_restore.set_defaults(func=cmd_snapshot_restore)

    ledger_save = subparsers.add_parser("ledger-save", help="append a diagnosis session to the ledger")
    _add_source_args(ledger_save)
    ledger_save.add_argument("--db", default=DEFAULT_DB)
    ledger_save.add_argument(
        "--inject",
        choices=("none", "wire-corrupt", "event-version-conflict"),
        default="none",
    )
    ledger_save.set_defaults(func=cmd_ledger_save)

    ledger_list = subparsers.add_parser("ledger-list", help="list ledger records")
    ledger_list.add_argument("--db", default=DEFAULT_DB)
    ledger_list.set_defaults(func=cmd_ledger_list)

    ledger_verify = subparsers.add_parser("ledger-verify", help="replay and compare a ledger record")
    ledger_verify.add_argument("--db", default=DEFAULT_DB)
    ledger_verify.add_argument("--ledger-id", required=True)
    ledger_verify.add_argument("--wire")
    ledger_verify.add_argument("--manifest")
    ledger_verify.add_argument("--events", nargs="+")
    ledger_verify.add_argument("--window-ms", type=int)
    ledger_verify.add_argument("--out")
    ledger_verify.set_defaults(func=cmd_ledger_verify)

    echo_serve = subparsers.add_parser("echo-serve", help="start the local loopback echo peer")
    echo_serve.add_argument("--port", type=int, required=True)
    echo_serve.add_argument("--state-db", default=DEFAULT_ECHO_DB)
    echo_serve.add_argument("--once", action="store_true")
    echo_serve.add_argument("--ack-timeout", type=float, default=5.0)
    echo_serve.set_defaults(func=cmd_echo_serve)

    echo_replay = subparsers.add_parser("echo-replay", help="replay diagnosis gaps through the loopback peer")
    _add_source_args(echo_replay)
    echo_replay.add_argument("--port", type=int, required=True)
    echo_replay.add_argument("--request-id")
    echo_replay.add_argument("--no-ack", action="store_true")
    echo_replay.add_argument("--out")
    echo_replay.add_argument(
        "--inject",
        choices=("none", "echo-mismatch"),
        default="none",
    )
    echo_replay.set_defaults(func=cmd_echo_replay)

    echo_recover = subparsers.add_parser("echo-recover", help="recover only persisted loopback peer state")
    echo_recover.add_argument("--state-db", default=DEFAULT_ECHO_DB)
    echo_recover.add_argument("--out")
    echo_recover.set_defaults(func=cmd_echo_recover)

    generate = subparsers.add_parser("generate-samples", help="regenerate deterministic samples")
    generate.set_defaults(func=cmd_generate_samples)

    gui = subparsers.add_parser("gui", help="open the Tkinter workbench")
    gui.add_argument("--db", default=DEFAULT_DB)
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        if args.command not in {"diagnose", "snapshot-save", "snapshot-restore", "echo-replay", "echo-recover"}:
            _emit(result, compact=args.compact)
        else:
            _emit(result, compact=args.compact)
        return 0
    except DiagnosticError as exc:
        _emit(exc.to_dict(), compact=args.compact)
        return 2
    except BrokenPipeError:
        return 1
    except Exception as exc:  # keep the CLI contract even for unexpected failures
        if os.environ.get("LOSS_DIAG_DEBUG"):
            raise
        _emit(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": str(exc),
                "details": {"exception": type(exc).__name__},
            },
            compact=args.compact,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

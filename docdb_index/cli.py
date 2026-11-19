"""Command-line interface for the local document range-index component."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json, loads_strict
from .engine import apply_changes, diagnose, initialize_session, manifest, query_index
from .errors import (
    DocumentError,
    ERROR_EXIT_CODE,
    DocDBError,
    StateError,
)
from .inject import inject_anomaly
from .replay import replay_validate
from .snapshot import (
    create_snapshot,
    list_snapshots,
    load_snapshot_manifest,
    restore_snapshot,
    snapshot_state,
)
from .storage import log_event
from .validation import load_declarations, read_json_file
from .verify import verify_session


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(canonical_json(payload) + "\n")


def _parse_number(text: str) -> int | float:
    try:
        value = loads_strict(text)
    except Exception as exc:
        raise DocumentError("value must be a finite number", field_path="$") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DocumentError("value must be a finite number", field_path="$")
    return value


def _changes_payload(path: str | Path, batch_id: str | None) -> tuple[str | None, list[dict[str, Any]], str]:
    root = read_json_file(path)
    if isinstance(root, list):
        return None, root, batch_id or "batch"
    if not isinstance(root, dict):
        raise DocumentError("changes file must be an object or an array", field_path="$")
    collection = root.get("collection", root.get("collection_name"))
    batches = root.get("batches")
    if batches is not None:
        if not isinstance(batches, list) or not batches:
            raise DocumentError("batches must be a non-empty array", field_path="batches")
        selected: dict[str, Any] | None = None
        if batch_id is not None:
            for item in batches:
                if isinstance(item, dict) and item.get("id", item.get("batch_id")) == batch_id:
                    selected = item
                    break
            if selected is None:
                raise DocumentError(
                    "requested batch id was not found",
                    field_path="batches",
                    batch_id=batch_id,
                )
        elif len(batches) == 1 and isinstance(batches[0], dict):
            selected = batches[0]
        else:
            raise DocumentError(
                "multiple batches require --batch-id",
                field_path="batches",
            )
        if not isinstance(selected, dict):
            raise DocumentError("batch must be an object", field_path="batches")
        changes = selected.get(
            "changes",
            selected.get("mutations", selected.get("operations")),
        )
        if not isinstance(changes, list):
            raise DocumentError("batch changes must be an array", field_path="batches.changes")
        selected_collection = selected.get("collection", collection)
        selected_id = selected.get("id", selected.get("batch_id", batch_id or "batch"))
        return (
            selected_collection if isinstance(selected_collection, str) else collection,
            changes,
            str(selected_id),
        )
    changes = root.get("changes", root.get("mutations", root.get("operations")))
    if not isinstance(changes, list):
        raise DocumentError("changes must be an array", field_path="changes")
    return collection if isinstance(collection, str) else None, changes, batch_id or "batch"


def _resolve_snapshot_path(session: str, value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    return Path(session) / "snapshots" / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m docdb_index",
        description="Local deterministic segmented range index over JSON document collections.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="validate a declaration and build a fresh session")
    init.add_argument("--session", required=True)
    init.add_argument("--input", required=True)
    init.add_argument("--reset", action="store_true")

    apply_parser = sub.add_parser("apply", help="atomically apply one change batch")
    apply_parser.add_argument("--session", required=True)
    apply_parser.add_argument("--input", required=True)
    apply_parser.add_argument("--collection")
    apply_parser.add_argument("--batch-id")
    apply_parser.add_argument("--batch-size", type=int)
    apply_parser.add_argument("--retain-stale", action="store_true")

    reclaim = sub.add_parser("reclaim", help="reclaim stale index entries without changing versions")
    reclaim.add_argument("--session", required=True)
    reclaim.add_argument("--collection", required=True)

    query = sub.add_parser("query", help="range/equality/ordered pagination query")
    query.add_argument("--session", required=True)
    query.add_argument("--collection", required=True)
    query.add_argument("--index", dest="index_path", required=True)
    choice = query.add_mutually_exclusive_group()
    choice.add_argument("--eq")
    choice.add_argument("--range", nargs=2, metavar=("LO", "HI"))
    query.add_argument("--lo")
    query.add_argument("--hi")
    query.add_argument("--offset", type=int, default=0)
    query.add_argument("--limit", type=int)
    query.add_argument("--scan", choices=("index", "full"), default="index")

    full = sub.add_parser("scan-full", help="reference full-document scan")
    full.add_argument("--session", required=True)
    full.add_argument("--collection", required=True)
    full.add_argument("--index", dest="index_path", required=True)
    choice = full.add_mutually_exclusive_group()
    choice.add_argument("--eq")
    choice.add_argument("--range", nargs=2, metavar=("LO", "HI"))
    full.add_argument("--lo")
    full.add_argument("--hi")
    full.add_argument("--offset", type=int, default=0)
    full.add_argument("--limit", type=int)

    diag = sub.add_parser("diagnose", help="stable per-segment stale/isolation report")
    diag.add_argument("--session", required=True)
    diag.add_argument("--collection", required=True)
    diag.add_argument("--index", dest="index_path", required=True)

    verify = sub.add_parser("verify", help="check segments, ledger and replay journal")
    verify.add_argument("--session", required=True)
    verify.add_argument("--no-replay", action="store_true")

    replay = sub.add_parser("replay", help="replay audit batches into a fresh session")
    replay.add_argument("--session", required=True)

    inject = sub.add_parser("inject", help="inject one deterministic business anomaly")
    inject.add_argument("--session", required=True)
    inject.add_argument("--collection", required=True)
    inject.add_argument("--index", dest="index_path", required=True)
    inject.add_argument(
        "--type",
        required=True,
        choices=("segment", "segment_digest", "digest", "bound", "boundary", "ledger", "ledger_missing"),
    )
    inject.add_argument("--segment-id")

    snapshot = sub.add_parser("snapshot", help="create/list/restore whole-session snapshots")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snapshot_create = snapshot_sub.add_parser("create")
    snapshot_create.add_argument("--session", required=True)
    snapshot_create.add_argument("--label")
    snapshot_list = snapshot_sub.add_parser("list")
    snapshot_list.add_argument("--session", required=True)
    snapshot_restore = snapshot_sub.add_parser("restore")
    snapshot_restore.add_argument("--session", required=True)
    snapshot_restore.add_argument("--snapshot", required=True)

    state = sub.add_parser("manifest", help="print deterministic state manifest")
    state.add_argument("--session", required=True)

    state_snapshot = sub.add_parser("snapshot-state", help="print current snapshot dimensions")
    state_snapshot.add_argument("--session", required=True)

    gui = sub.add_parser("gui", help="Tkinter diagnostic interface")
    gui.add_argument("--session", required=True)
    gui.add_argument("--collection")
    gui.add_argument("--index", dest="index_path")

    return parser


def _command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "init":
        specs = load_declarations(args.input)
        return initialize_session(args.session, specs, reset=args.reset)

    if args.command == "apply":
        collection, changes, batch_id = _changes_payload(args.input, args.batch_id)
        collection = args.collection or collection
        if not collection:
            raise DocumentError("collection is required", field_path="collection")
        return apply_changes(
            args.session,
            collection,
            changes,
            batch_id=batch_id,
            batch_size=args.batch_size,
            retain_stale=args.retain_stale,
        )

    if args.command == "reclaim":
        return apply_changes(
            args.session,
            args.collection,
            [],
            batch_id="reclaim",
            retain_stale=False,
        )

    if args.command in {"query", "scan-full"}:
        if (args.lo is not None or args.hi is not None) and (
            args.eq is not None or args.range is not None
        ):
            raise DocumentError(
                "--lo/--hi cannot be combined with --eq or --range",
                field_path="range",
            )
        eq = None if args.eq is None else _parse_number(args.eq)
        lo = hi = None
        if args.range:
            lo = _parse_number(args.range[0])
            hi = _parse_number(args.range[1])
        elif args.lo is not None or args.hi is not None:
            lo = None if args.lo is None else _parse_number(args.lo)
            hi = None if args.hi is None else _parse_number(args.hi)
        scan = "full" if args.command == "scan-full" else args.scan
        return query_index(
            args.session,
            args.collection,
            args.index_path,
            eq=eq,
            lo=lo,
            hi=hi,
            offset=args.offset,
            limit=args.limit,
            scan=scan,
        )

    if args.command == "diagnose":
        return diagnose(args.session, args.collection, args.index_path)

    if args.command == "verify":
        return verify_session(args.session, include_replay=not args.no_replay)

    if args.command == "replay":
        return replay_validate(args.session)

    if args.command == "inject":
        return inject_anomaly(
            args.session,
            collection_name=args.collection,
            index_path=args.index_path,
            kind=args.type,
            segment_id=args.segment_id,
        )

    if args.command == "snapshot":
        if args.snapshot_command == "create":
            return create_snapshot(args.session, label=args.label)
        if args.snapshot_command == "list":
            return {"ok": True, "snapshots": list_snapshots(args.session)}
        if args.snapshot_command == "restore":
            return restore_snapshot(
                args.session,
                _resolve_snapshot_path(args.session, args.snapshot),
            )
        raise StateError("unknown snapshot command")

    if args.command == "manifest":
        return manifest(args.session)

    if args.command == "snapshot-state":
        return {"ok": True, "state": snapshot_state(args.session)}

    if args.command == "gui":
        try:
            from .gui import run_gui
        except ModuleNotFoundError as exc:
            raise StateError(
                "Tkinter is not available in this Python installation",
                module=exc.name,
            ) from exc
        run_gui(args.session, collection=args.collection, index_path=args.index_path)
        return {"ok": True, "gui": "closed"}

    raise StateError("unknown command", command=args.command)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    session = getattr(args, "session", None)
    try:
        payload = _command(args)
        if payload.get("ok") is not False:
            emit(payload)
        return 0
    except DocDBError as exc:
        if session and not getattr(exc, "_docdb_logged", False):
            log_event(
                session,
                level="error",
                code=exc.code,
                message=exc.message,
                context=exc.details,
            )
        emit(exc.payload())
        return ERROR_EXIT_CODE
    except Exception as exc:
        if session:
            log_event(
                session,
                level="error",
                code="E_INTERNAL",
                message=str(exc),
                context={"type": type(exc).__name__},
            )
        emit(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": str(exc),
                "type": type(exc).__name__,
            }
        )
        return ERROR_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())

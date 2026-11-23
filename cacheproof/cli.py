"""Command-line interface for cacheproof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .canonical import canonical_text
from .diagnostics import (
    dashboard_snapshot,
    inject_anomaly,
    render_text_dashboard,
)
from .errors import DiagnosticError
from .eventlog import EventLogger
from .paths import WorkspacePaths, reset_demo_workspace
from .spec import load_spec
from .store import CacheProofStore


DEFAULT_WORKSPACE = ".cacheproof"


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _emit(args: argparse.Namespace, value: Any) -> None:
    if getattr(args, "json", False):
        _json_print(value)
    else:
        print(canonical_text(value))


def _open_store(
    args: argparse.Namespace,
    *,
    auto_recover: bool = True,
    auto_snapshot: bool = False,
) -> CacheProofStore:
    return CacheProofStore(
        args.workspace,
        auto_recover=auto_recover,
        auto_snapshot=auto_snapshot,
    )


def _log_cli_error(args: argparse.Namespace, error: DiagnosticError) -> None:
    workspace = getattr(args, "workspace", DEFAULT_WORKSPACE)
    try:
        paths = WorkspacePaths(workspace)
        logger = EventLogger(paths.event_log)
        operation_seq = None
        callback_sequence = error.details.get("operation_seq")
        if isinstance(callback_sequence, int) and not isinstance(
            callback_sequence, bool
        ):
            operation_seq = callback_sequence
        logger.emit(
            "business_error",
            business_code=error.code,
            operation_seq=operation_seq,
            segment_id=error.segment_id,
            entry_key=error.entry_key,
            field=error.field,
            details=error.to_dict(),
            monotonic_ns=error.monotonic_ns,
        )
    except Exception:
        pass


def _command_init(args: argparse.Namespace) -> dict[str, Any]:
    if args.reset:
        paths = WorkspacePaths(args.workspace)
        reset_demo_workspace(paths)
    spec = load_spec(args.spec)
    store = CacheProofStore(
        args.workspace,
        spec=spec,
        create=True,
        auto_snapshot=False,
    )
    try:
        report = store.verify_all()
        snapshot = store.save_snapshot()
        return {
            "ok": True,
            "code": "OK",
            "workspace": str(store.paths.root),
            "spec_digest": spec_digest_cacheable(spec),
            "ledger": report["ledger"],
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "capacity_bytes": segment["capacity_bytes"],
                    "used_bytes": segment["used_bytes"],
                    "remaining_bytes": segment["remaining_bytes"],
                    "proof_content_digest": segment["proof_content_digest"],
                }
                for segment in report["segments"]
            ],
            "snapshot_digest": snapshot["content_digest"],
        }
    finally:
        store.close()


def spec_digest_cacheable(spec: Any) -> str:
    from .spec import spec_digest

    return spec_digest(spec)


def _command_show(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args)
    try:
        dashboard = dashboard_snapshot(store)
        if not args.json:
            print(render_text_dashboard(dashboard))
            return dashboard
        return dashboard
    finally:
        store.close()


def _command_verify(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args)
    try:
        return store.verify_all()
    finally:
        store.close()


def _command_put(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args)
    try:
        result = store.put_entry(
            args.segment, args.key, args.size, args.seed
        ).to_dict()
        result["snapshot"] = store.save_snapshot()
        return result
    finally:
        store.close()


def _command_delete(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args)
    try:
        result = store.delete_entry(args.segment, args.key).to_dict()
        result["snapshot"] = store.save_snapshot()
        return result
    finally:
        store.close()


def _command_view(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args, auto_snapshot=True)
    try:
        return store.get_view(args.view_id, rebuild=not args.preview)
    finally:
        store.close()


def _command_recover(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args, auto_recover=False)
    try:
        result = store.recover_unknown_entries()
        if result["newly_annotated"] or result["prepared_operations_seen"]:
            result["snapshot"] = store.save_snapshot()
        return result
    finally:
        store.close()


def _command_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args, auto_recover=True, auto_snapshot=False)
    try:
        if args.snapshot_action == "save":
            return store.save_snapshot()
        if args.snapshot_action == "restore":
            return store.restore_snapshot()
        raise DiagnosticError(
            "E_SPEC", "unknown snapshot action", field="snapshot_action"
        )
    finally:
        store.close()


def _command_ui(args: argparse.Namespace) -> dict[str, Any]:
    from .ui import launch_dashboard

    store = _open_store(args)
    try:
        launch_dashboard(store)
        return {"ok": True, "code": "OK", "closed": True}
    finally:
        store.close()


def _command_diagnose(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args)
    try:
        inject_anomaly(
            store,
            args.inject,
            segment_id=args.segment,
            entry_key=args.key,
        )
        raise DiagnosticError(
            "E_IO", "injection unexpectedly succeeded", field="inject"
        )
    finally:
        store.close()


def _command_crash_worker(args: argparse.Namespace) -> dict[str, Any]:
    store = _open_store(args)
    try:
        store.put_entry(
            args.segment,
            args.key,
            args.size,
            args.seed,
            failpoint=args.failpoint,
        )
        return {
            "ok": False,
            "code": "E_IO",
            "message": "crash failpoint did not terminate the process",
        }
    finally:
        store.close()


def _command_demo(args: argparse.Namespace) -> dict[str, Any]:
    from .demo import run_demo

    return run_demo(
        args.workspace,
        reset=args.reset,
        report_path=Path(args.report) if args.report else None,
    )


def _add_common(parser: argparse.ArgumentParser, *, workspace_default: str = DEFAULT_WORKSPACE) -> None:
    parser.add_argument("--workspace", default=argparse.SUPPRESS)
    parser.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="emit structured JSON"
    )
    parser.set_defaults(command_workspace_default=workspace_default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cacheproof",
        description="Cache-segment capacity proof and read-view convergence diagnostics",
    )
    parser.add_argument("--workspace", dest="global_workspace", default=None)
    parser.add_argument("--json", dest="global_json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="initialize a workspace from JSON")
    _add_common(init_parser)
    init_parser.add_argument("--spec", required=True)
    init_parser.add_argument("--reset", action="store_true")
    init_parser.set_defaults(handler=_command_init)

    show_parser = subparsers.add_parser("show", help="render the diagnostic dashboard")
    _add_common(show_parser)
    show_parser.set_defaults(handler=_command_show)

    verify_parser = subparsers.add_parser("verify", help="verify ledger and proofs")
    _add_common(verify_parser)
    verify_parser.set_defaults(handler=_command_verify)

    put_parser = subparsers.add_parser("put", help="write a deterministic entry")
    _add_common(put_parser)
    put_parser.add_argument("--segment", required=True)
    put_parser.add_argument("--key", required=True)
    put_parser.add_argument("--size", required=True, type=int)
    put_parser.add_argument("--seed", required=True)
    put_parser.set_defaults(handler=_command_put)

    delete_parser = subparsers.add_parser("delete", help="delete an entry")
    _add_common(delete_parser)
    delete_parser.add_argument("--segment", required=True)
    delete_parser.add_argument("--key", required=True)
    delete_parser.set_defaults(handler=_command_delete)

    view_parser = subparsers.add_parser("view", help="query or preview a read view")
    _add_common(view_parser)
    view_parser.add_argument("--view-id", required=True)
    view_parser.add_argument("--preview", action="store_true")
    view_parser.set_defaults(handler=_command_view)

    recover_parser = subparsers.add_parser(
        "recover", help="annotate interrupted writes as unknown"
    )
    _add_common(recover_parser)
    recover_parser.set_defaults(handler=_command_recover)

    snapshot_parser = subparsers.add_parser("snapshot", help="save or restore a session snapshot")
    _add_common(snapshot_parser)
    snapshot_parser.add_argument("snapshot_action", choices=("save", "restore"))
    snapshot_parser.set_defaults(handler=_command_snapshot)

    ui_parser = subparsers.add_parser("ui", help="open the Tkinter dashboard")
    _add_common(ui_parser)
    ui_parser.set_defaults(handler=_command_ui)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="inject one supported anomaly and report its business code"
    )
    _add_common(diagnose_parser)
    diagnose_parser.add_argument(
        "--inject", required=True, choices=("truncate", "capacity", "ledger-digest")
    )
    diagnose_parser.add_argument("--segment")
    diagnose_parser.add_argument("--key")
    diagnose_parser.set_defaults(handler=_command_diagnose)

    crash_parser = subparsers.add_parser(
        "_crash-worker", help="internal crash failpoint worker"
    )
    _add_common(crash_parser)
    crash_parser.add_argument("--segment", required=True)
    crash_parser.add_argument("--key", required=True)
    crash_parser.add_argument("--size", required=True, type=int)
    crash_parser.add_argument("--seed", required=True)
    crash_parser.add_argument(
        "--failpoint", required=True, choices=("after_prepare", "after_file_replace", "after_commit")
    )
    crash_parser.set_defaults(handler=_command_crash_worker)

    demo_parser = subparsers.add_parser("demo", help="run the complete repeatable demo")
    _add_common(demo_parser, workspace_default="demo-workspace")
    demo_parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reset the demo workspace before running (default: true)",
    )
    demo_parser.add_argument("--report")
    demo_parser.set_defaults(handler=_command_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "workspace"):
        args.workspace = (
            getattr(args, "global_workspace", None)
            or getattr(args, "command_workspace_default", DEFAULT_WORKSPACE)
        )
    if not hasattr(args, "json"):
        args.json = getattr(args, "global_json", False)
    try:
        result = args.handler(args)
        if result is not None and getattr(args, "json", False):
            _json_print(result)
        elif result is not None and args.command in {
            "init",
            "verify",
            "put",
            "delete",
            "view",
            "recover",
            "snapshot",
            "diagnose",
            "demo",
        }:
            print(canonical_text(result))
        return 0
    except DiagnosticError as exc:
        _log_cli_error(args, exc)
        if getattr(args, "json", False):
            _json_print(exc.to_dict())
        else:
            location = []
            if exc.segment_id is not None:
                location.append(f"segment={exc.segment_id}")
            if exc.entry_key is not None:
                location.append(f"entry={exc.entry_key}")
            if exc.field is not None:
                location.append(f"field={exc.field}")
            suffix = " " + " ".join(location) if location else ""
            print(f"{exc.code}: {exc.message}{suffix}", file=sys.stderr)
            if exc.details:
                print(canonical_text(exc.details), file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        error = DiagnosticError(
            "E_IO", f"operating-system error: {exc}", field="filesystem"
        )
        _log_cli_error(args, error)
        if getattr(args, "json", False):
            _json_print(error.to_dict())
        else:
            print(f"{error.code}: {error.message}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())

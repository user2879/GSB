"""Command-line interface for validation, execution, evolution and recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import (
    E_DIVERGE,
    E_HANDSHAKE,
    E_MANIFEST,
    E_PORT,
    E_REPLY,
    E_REPLAY,
    PipelineError,
)
from .evolution import build_evolution
from .manifest import load_manifest
from .planner import compare_plans, compile_plan
from .session import PipelineSession
from .store import Store
from .util import canonical_json


EXIT_CODES = {
    E_MANIFEST: 2,
    E_PORT: 3,
    E_HANDSHAKE: 4,
    E_REPLY: 5,
    E_DIVERGE: 6,
    E_REPLAY: 7,
}


def _emit(value: Any, *, stream: Any = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(canonical_json(value) + "\n")
    stream.flush()


def _default_snapshot(db_path: str) -> str:
    return f"{db_path}.snapshot.json"


def _session(
    manifest_path: str,
    db_path: str,
    snapshot_path: str | None,
    *,
    handshake_timeout: float,
    dispatch_timeout: float,
    scheduler_timeout: float,
) -> PipelineSession:
    return PipelineSession(
        load_manifest(manifest_path),
        db_path,
        snapshot_path=snapshot_path or _default_snapshot(db_path),
        handshake_timeout=handshake_timeout,
        dispatch_timeout=dispatch_timeout,
        scheduler_timeout=scheduler_timeout,
    )


def _report_json(report: Any) -> dict[str, Any]:
    payload = report.canonical()
    payload["ok"] = not report.failed_shards
    return payload


def _cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    plan = compile_plan(manifest)
    _emit(
        {
            "ok": True,
            "manifest_digest": manifest.manifest_digest,
            "cluster_version": manifest.cluster_version,
            "revision_chain_version": manifest.revision_chain_version,
            "budget_version": manifest.budget_version,
            "latest_revision": manifest.latest_revision.version,
            "clusters": len(manifest.clusters),
            "tasks": len(manifest.tasks),
            "revisions": len(manifest.revisions),
            "shards": len(plan.shards),
            "plan_digest": plan.plan_digest,
        }
    )
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    plan = compile_plan(manifest)
    text = plan.serialized()
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    session = _session(
        args.manifest,
        args.db,
        args.snapshot,
        handshake_timeout=args.handshake_timeout,
        dispatch_timeout=args.dispatch_timeout,
        scheduler_timeout=args.scheduler_timeout,
    )
    try:
        report = session.run_current(
            verify_replay=args.verify_replay,
            crash_after_completed=args.crash_after_completed,
        )
        payload = _report_json(report)
        _emit(payload)
        return 0 if payload["ok"] else EXIT_CODES[E_REPLY]
    finally:
        session.close()


def _cmd_recover(args: argparse.Namespace) -> int:
    session = _session(
        args.manifest,
        args.db,
        args.snapshot,
        handshake_timeout=args.handshake_timeout,
        dispatch_timeout=args.dispatch_timeout,
        scheduler_timeout=args.scheduler_timeout,
    )
    try:
        report = session.recover(crash_after_completed=args.crash_after_completed)
        payload = _report_json(report)
        _emit(payload)
        return 0 if payload["ok"] else EXIT_CODES[E_REPLY]
    finally:
        session.close()


def _cmd_evolve(args: argparse.Namespace) -> int:
    old = load_manifest(args.old_manifest)
    new = load_manifest(args.manifest)
    session = PipelineSession(
        old,
        args.db,
        snapshot_path=args.snapshot or _default_snapshot(args.db),
        handshake_timeout=args.handshake_timeout,
        dispatch_timeout=args.dispatch_timeout,
        scheduler_timeout=args.scheduler_timeout,
    )
    try:
        report = session.evolve(
            new,
            verify_replay=args.verify_replay,
            crash_after_completed=args.crash_after_completed,
        )
        payload = _report_json(report)
        _emit(payload)
        return 0 if payload["ok"] else EXIT_CODES[E_REPLY]
    finally:
        session.close()


def _cmd_replay(args: argparse.Namespace) -> int:
    session = _session(
        args.manifest,
        args.db,
        args.snapshot,
        handshake_timeout=args.handshake_timeout,
        dispatch_timeout=args.dispatch_timeout,
        scheduler_timeout=args.scheduler_timeout,
    )
    try:
        _emit(session.replay())
        return 0
    finally:
        session.close()


def _cmd_compare(args: argparse.Namespace) -> int:
    left_manifest = load_manifest(args.left)
    right_manifest = load_manifest(args.right)
    left = compile_plan(left_manifest)
    right = compile_plan(right_manifest)
    compare_plans(left, right)
    _emit(
        {
            "ok": True,
            "plan_digest": left.plan_digest,
            "shards": len(left.shards),
            "byte_identical": left.serialized() == right.serialized(),
        }
    )
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    old = load_manifest(args.old_manifest)
    new = load_manifest(args.manifest)
    delta = build_evolution(old, new)
    _emit({"ok": True, **delta.canonical()})
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        snapshot = store.latest_snapshot()
        payload = {
            "ok": True,
            "db": str(Path(args.db).resolve()),
            "ledger_cursor": store.ledger_cursor,
            "ledger_format_version": store.ledger_format_version,
            "ledger_content_digest": store.ledger_content_digest,
            "ledger_entries": len(store.ledger_entries()),
            "snapshot": None,
            "current_view": list(store.current_view()),
            "current_invalidations": list(store.current_invalidations()),
            "current_stats": store.current_stats(),
        }
        if snapshot is not None:
            payload["snapshot"] = {
                "generation": snapshot["generation"],
                "format_version": snapshot["format_version"],
                "content_digest": snapshot["content_digest"],
                "cluster_version": snapshot["cluster_version"],
                "revision_chain_version": snapshot["revision_chain_version"],
                "budget_version": snapshot["budget_version"],
                "ledger_cursor": snapshot["ledger_cursor"],
                "phase": snapshot["payload"].get("phase"),
                "latest_revision": snapshot["payload"].get("latest_revision"),
                "diff": snapshot["payload"].get("diff"),
            }
        _emit(payload)
        return 0
    finally:
        store.close()


def _cmd_ui(args: argparse.Namespace) -> int:
    from .ui import launch_ui

    return launch_ui(args.db)


def _common_session_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="SQLite ledger/snapshot database path")
    parser.add_argument("--snapshot", help="snapshot JSON path (defaults beside the DB)")
    parser.add_argument("--handshake-timeout", type=float, default=5.0)
    parser.add_argument("--dispatch-timeout", type=float, default=30.0)
    parser.add_argument("--scheduler-timeout", type=float, default=120.0)
    parser.add_argument(
        "--crash-after-completed",
        type=int,
        metavar="N",
        help="test hook: leave the next shard in-flight and exit 137 after N commits",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="config-pipeline",
        description="Deterministic local multi-cluster configuration distribution pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strongly validate a manifest")
    validate.add_argument("manifest")
    validate.set_defaults(handler=_cmd_validate)

    compile_parser = subparsers.add_parser("compile", help="compile a deterministic shard plan")
    compile_parser.add_argument("manifest")
    compile_parser.add_argument("--output", help="also write canonical plan bytes here")
    compile_parser.set_defaults(handler=_cmd_compile)

    run = subparsers.add_parser("run", help="run or resume the current manifest")
    run.add_argument("manifest")
    _common_session_arguments(run)
    run.add_argument("--verify-replay", action="store_true")
    run.set_defaults(handler=_cmd_run)

    recover = subparsers.add_parser("recover", help="replay committed shards then finish remaining work")
    recover.add_argument("manifest")
    _common_session_arguments(recover)
    recover.set_defaults(handler=_cmd_recover)

    evolve = subparsers.add_parser("evolve", help="apply a monotonic revision evolution")
    evolve.add_argument("old_manifest")
    evolve.add_argument("manifest")
    _common_session_arguments(evolve)
    evolve.add_argument("--verify-replay", action="store_true")
    evolve.set_defaults(handler=_cmd_evolve)

    replay = subparsers.add_parser("replay", help="recompute committed aggregates and compare the ledger")
    replay.add_argument("manifest")
    _common_session_arguments(replay)
    replay.set_defaults(handler=_cmd_replay)

    compare = subparsers.add_parser("compare", help="compare two compiled plans")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.set_defaults(handler=_cmd_compare)

    diff = subparsers.add_parser("diff", help="show invalidation propagation between two manifests")
    diff.add_argument("old_manifest")
    diff.add_argument("manifest")
    diff.set_defaults(handler=_cmd_diff)

    status = subparsers.add_parser("status", help="show the current snapshot and ledger status")
    status.add_argument("--db", required=True)
    status.set_defaults(handler=_cmd_status)

    ui = subparsers.add_parser("ui", help="open a read-only Tkinter status viewer")
    ui.add_argument("--db", required=True)
    ui.set_defaults(handler=_cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PipelineError as exc:
        _emit({"ok": False, **exc.to_dict()})
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        _emit({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"})
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

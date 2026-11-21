"""Command line entry point and fault injection switches.

Injection only changes input files or parameters, never source code:
  * --inject-manifest  rewrites the manifest into invalid JSON -> E_MANIFEST
  * --inject-port      pre-binds a node port before launch        -> E_PORT
  * --inject-version   mutates the stored snapshot version/hash    -> E_VERSION
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

from .errors import LeaseError, ManifestError, PortInUseError, VersionConflictError
from .manifest import Revision, load_manifest
from .platform import NodeSpec, Platform, PlatformConfig
from .storage import Store
from .structured_log import StructuredLogger


def _load_node_registry(path: str | Path) -> list[NodeSpec]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [NodeSpec(node_id=n["node_id"], port=int(n["port"]),
                     host=n.get("host", "127.0.0.1")) for n in doc["nodes"]]


def _load_requests(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["requests"]


def _inject_manifest_corruption(path: str | Path) -> None:
    p = Path(path)
    backup = p.with_suffix(p.suffix + ".bak")
    backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    # Truncated JSON: deterministic parse failure, file input only.
    p.write_text('{"revisions": [ {"rev_seq": 1, ', encoding="utf-8")


def _inject_snapshot_conflict(db_path: str, snapshot: str) -> None:
    store = Store(db_path)
    snap = store.load_snapshot(snapshot)
    if snap is None:
        store.close()
        raise VersionConflictError("snapshot not found for injection", snapshot=snapshot)
    store.save_snapshot(
        snapshot, snap["version"] + 999,
        "0" * 64, snap["state"], snap["updated_ms"],
    )
    store.close()


async def _run_serve(args: argparse.Namespace) -> int:
    runtime = Path(args.workdir)
    runtime.mkdir(parents=True, exist_ok=True)
    db_path = runtime / "platform.sqlite3"
    log_dir = runtime / "logs"
    session_id = args.session or f"sess-{uuid.uuid4().hex[:8]}"

    if args.inject_manifest:
        _inject_manifest_corruption(args.manifest)

    logger = StructuredLogger(log_dir, session_id)
    try:
        revisions = load_manifest(args.manifest)
    except ManifestError as exc:
        logger.error(exc.code, message=exc.message, **exc.context)
        logger.close()
        print(json.dumps({"status": "error", **exc.to_dict()}, ensure_ascii=False))
        return 2

    node_specs = _load_node_registry(args.nodes)
    node_budgets = {n.node_id: args.node_budget for n in node_specs}
    config = PlatformConfig(
        db_path=str(db_path), log_dir=str(log_dir), session_id=session_id,
        global_budget=args.global_budget, node_budgets=node_budgets,
        lease_ttl_ms=args.lease_ttl_ms,
    )
    store = Store(db_path)
    for rev in revisions:
        store.upsert_revision(rev.to_dict())
    platform = Platform(config, revisions, node_specs, store=store, logger=logger)

    holders: list[socket.socket] = []
    if args.inject_port and node_specs:
        target = node_specs[0]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((target.host, target.port))
        sock.listen(1)
        holders.append(sock)

    try:
        await platform.start_nodes()
        requests = _load_requests(args.requests)
        results = await platform.dispatch_many(requests)
        snapshot = platform.save_snapshot("session")
        report = {
            "status": "ok",
            "session_id": session_id,
            "applied": results,
            "snapshot_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
            "budgets": platform.budgets.snapshots(),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except LeaseError as exc:
        platform._log_error(exc)
        print(json.dumps({"status": "error", **exc.to_dict()}, ensure_ascii=False))
        return 2
    finally:
        for sock in holders:
            sock.close()
        await platform.close()


async def _run_recover(args: argparse.Namespace) -> int:
    runtime = Path(args.workdir)
    db_path = runtime / "platform.sqlite3"
    log_dir = runtime / "logs"
    session_id = args.session or f"recover-{uuid.uuid4().hex[:8]}"
    store = Store(db_path)
    revisions = [Revision(r["rev_seq"], r["parent"], r["changes"],
                          r.get("kind", "normal"), r.get("rollback_target"))
                 for r in store.list_revisions()]
    node_specs = _load_node_registry(args.nodes)
    config = PlatformConfig(
        db_path=str(db_path), log_dir=str(log_dir), session_id=session_id,
    )
    platform = Platform(config, revisions, node_specs, store=store,
                        logger=StructuredLogger(log_dir, session_id))
    recovery = platform.recover()
    await platform.start_nodes()
    acks = await platform.rebuild_nodes_after_recover()
    report = {**recovery, "rebuilt_nodes": [a["node_id"] for a in acks]}
    try:
        report["exit_report"] = platform.build_exit_report()
    except LeaseError as exc:
        report["exit_report_blocked"] = exc.to_dict()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    await platform.close()
    return 0


async def _run_reopen(args: argparse.Namespace) -> int:
    runtime = Path(args.workdir)
    db_path = runtime / "platform.sqlite3"
    log_dir = runtime / "logs"
    session_id = args.session or f"reopen-{uuid.uuid4().hex[:8]}"
    if args.inject_version:
        _inject_snapshot_conflict(str(db_path), args.snapshot)
    store = Store(db_path)
    revisions = [Revision(r["rev_seq"], r["parent"], r["changes"],
                          r.get("kind", "normal"), r.get("rollback_target"))
                 for r in store.list_revisions()]
    node_specs = _load_node_registry(args.nodes)
    config = PlatformConfig(
        db_path=str(db_path), log_dir=str(log_dir), session_id=session_id,
    )
    platform = Platform(config, revisions, node_specs, store=store,
                        logger=StructuredLogger(log_dir, session_id))
    try:
        result = platform.reopen_with_snapshot(args.snapshot)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except VersionConflictError as exc:
        platform._log_error(exc)
        print(json.dumps({"status": "error", **exc.to_dict()}, ensure_ascii=False))
        return 2
    finally:
        await platform.close()


def _run_logs(args: argparse.Namespace) -> int:
    store = Store(str(Path(args.workdir) / "platform.sqlite3"))
    rows = store.query_events(node_id=args.node, rev_seq=args.rev, code=args.code)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="configlease",
                                     description="lease distribution & budget arbitration")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run a full dispatch session")
    serve.add_argument("--workdir", required=True)
    serve.add_argument("--manifest", required=True)
    serve.add_argument("--nodes", required=True)
    serve.add_argument("--requests", required=True)
    serve.add_argument("--session")
    serve.add_argument("--global-budget", type=int, default=8)
    serve.add_argument("--node-budget", type=int, default=4)
    serve.add_argument("--lease-ttl-ms", type=int, default=10_000)
    serve.add_argument("--inject-manifest", action="store_true")
    serve.add_argument("--inject-port", action="store_true")
    serve.set_defaults(func=_run_serve)

    recover = sub.add_parser("recover", help="recover after forced termination")
    recover.add_argument("--workdir", required=True)
    recover.add_argument("--nodes", required=True)
    recover.add_argument("--session")
    recover.set_defaults(func=_run_recover)

    reopen = sub.add_parser("reopen", help="reopen snapshot and replay")
    reopen.add_argument("--workdir", required=True)
    reopen.add_argument("--nodes", required=True)
    reopen.add_argument("--snapshot", default="session")
    reopen.add_argument("--session")
    reopen.add_argument("--inject-version", action="store_true")
    reopen.set_defaults(func=_run_reopen)

    logs = sub.add_parser("logs", help="retrieve structured logs")
    logs.add_argument("--workdir", required=True)
    logs.add_argument("--node")
    logs.add_argument("--rev", type=int)
    logs.add_argument("--code")
    logs.set_defaults(func=_run_logs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if asyncio.iscoroutinefunction(args.func):
        return asyncio.run(args.func(args))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

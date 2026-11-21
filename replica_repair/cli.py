"""Command-line interface for local deterministic operation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .engine import run_pipeline
from .errors import BudgetError, EnvironmentError, RepairError, StateError
from .ledger import Ledger
from .manifest_repo import ManifestRepository, compute_state_digest
from .manifest_store import ManifestStore
from .model import Manifest
from .node_service import NodeClient, NodeServer
from .planner import RepairBudget
from .replay import ReplayVerifier
from .sample import (
    environment_report,
    init_sample,
    run_demo,
    start_servers,
)
from .util import atomic_write_bytes, canonical_json_dumps


EXIT_CODES = {
    "E_FORMAT": 2,
    "E_BLOCK": 3,
    "E_INTEGRITY": 4,
    "E_BUDGET": 5,
    "E_REPLAY": 6,
    "E_STATE": 7,
    "E_NETWORK": 8,
    "E_ENV": 9,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replica-repair",
        description="Local deterministic replica-manifest upgrade, truth, repair, ledger, and replay.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="create deterministic v1/v2/v3 samples and run the full local demo")
    demo.add_argument("--root", default=".", help="task root (default: current directory)")

    init = sub.add_parser("init", help="create deterministic sample directories and manifests")
    init.add_argument("--root", default=".")
    init.add_argument("--force", action="store_true")

    env = sub.add_parser("env", help="report Python, SQLite, NumPy, Tk and loopback-port readiness")

    validate = sub.add_parser("validate", help="strictly decode a manifest without upgrading")
    validate.add_argument("manifest")

    upgrade = sub.add_parser("upgrade", help="upgrade a legacy manifest in place with a resumable checkpoint")
    upgrade.add_argument("--root", default=".")
    upgrade.add_argument("manifest")
    upgrade.add_argument("--crash-after", type=int, default=0, help="forced-termination test hook (records committed)")

    serve = sub.add_parser("serve-node", help="serve one node directory over 127.0.0.1 JSON lines")
    serve.add_argument("--root", default=".")
    serve.add_argument("--manifest", required=True)
    serve.add_argument("--node", required=True)
    serve.add_argument("--port", type=int, default=0, help="0 chooses a free high loopback port")
    serve.add_argument("--port-file")

    run = sub.add_parser("run", help="run truth selection, budgeted repair, ledger write, and replay")
    run.add_argument("--root", default=".")
    run.add_argument("--manifest", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--max-parallel", type=int, default=3)
    run.add_argument("--per-node-kb", type=int, default=8)
    run.add_argument("--cost-per-kb", type=int, default=1)
    run.add_argument("--no-replay", action="store_true")
    run.add_argument("--node", action="append", default=[], metavar="NODE=HOST:PORT", help="use an existing node service")
    run.add_argument("--crash-after-task", type=int, default=0, help="forced-termination test hook after committed task")
    run.add_argument("--inject-truncate", action="append", default=[], metavar="NODE:BLOCK", help="truncate persisted block bytes")
    run.add_argument("--inject-corrupt", action="append", default=[], metavar="NODE:BLOCK", help="flip persisted block bytes")

    report = sub.add_parser("report", help="print the current ledger view; stale conclusions are excluded")
    report.add_argument("--ledger", required=True)

    replay = sub.add_parser("replay", help="recompute current digests/truth/plan and compare them to the ledger")
    replay.add_argument("--root", default=".")
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--ledger", required=True)

    evolve = sub.add_parser("evolve", help="change node priority and atomically advance manifest_version")
    evolve.add_argument("--manifest", required=True)
    evolve.add_argument("--set-priority", action="append", required=True, metavar="NODE=PRIORITY")

    inject = sub.add_parser("inject", help="inject a block truncation or same-size digest corruption")
    inject.add_argument("--root", default=".")
    inject.add_argument("kind", choices=["truncate", "corrupt"])
    inject.add_argument("target", metavar="NODE:BLOCK")

    view = sub.add_parser("view", help="optional Tk ledger viewer (requires tkinter)")
    view.add_argument("--ledger", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(Path(args.root))
            _print_json(result)
            return 0
        if args.command == "init":
            result = init_sample(Path(args.root), force=args.force)
            _print_json({"ok": True, **result})
            return 0
        if args.command == "env":
            _print_json({"ok": True, **environment_report()})
            return 0
        if args.command == "validate":
            manifest = Manifest.from_file(args.manifest)
            _print_json({"ok": True, "manifest": manifest.to_dict(), "fingerprint": manifest.fingerprint()})
            return 0
        if args.command == "upgrade":
            def hook(count: int) -> None:
                if args.crash_after > 0 and count >= args.crash_after:
                    os._exit(86)

            result = ManifestStore(args.root).upgrade_in_place(args.manifest, after_record=hook)
            _print_json({"ok": True, **result.to_dict()})
            return 0
        if args.command == "serve-node":
            server = NodeServer(args.root, args.manifest, args.node, port=args.port)
            if args.port_file:
                atomic_write_bytes(
                    Path(args.port_file),
                    canonical_json_dumps({"node_id": args.node, "host": server.host, "port": server.port}).encode("utf-8") + b"\n",
                )
            _print_json({"ok": True, "node_id": args.node, "host": server.host, "port": server.port})
            server.start()
            try:
                import threading

                threading.Event().wait()
            except KeyboardInterrupt:
                pass
            finally:
                server.stop()
            return 0
        if args.command == "run":
            result = _command_run(args)
            _print_json(result)
            return 0
        if args.command == "report":
            with Ledger(args.ledger) as ledger:
                recovery = ledger.reopen_from_disk()
                report = ledger.current_report()
                if recovery is not None:
                    report["reopen_check"] = recovery.to_dict()
                _print_json(report)
            return 0
        if args.command == "replay":
            result = _command_replay(args)
            _print_json(result)
            return 0
        if args.command == "evolve":
            result = _command_evolve(args)
            _print_json(result)
            return 0
        if args.command == "inject":
            result = _command_inject(args)
            _print_json(result)
            return 0
        if args.command == "view":
            try:
                from .tkview import show_ledger
                show_ledger(args.ledger)
            except ModuleNotFoundError as exc:
                raise EnvironmentError(
                    "tkinter is unavailable in this Python installation",
                    details={"module": "tkinter"},
                ) from exc
            return 0
    except RepairError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return EXIT_CODES.get(exc.code, 1)
    except OSError as exc:
        error = StateError("filesystem operation failed", details={"message": str(exc)})
        _print_json(error.to_dict(), stream=sys.stderr)
        return EXIT_CODES["E_STATE"]
    raise AssertionError("unhandled command")


def _command_run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    budget = RepairBudget(
        max_parallel=args.max_parallel,
        per_node_kb=args.per_node_kb,
        cost_per_kb=args.cost_per_kb,
    )
    budget.validate()
    upgraded = ManifestStore(root).load_current(args.manifest)
    manifest = upgraded.manifest
    _apply_injections(root, manifest, args.inject_truncate, "truncate")
    _apply_injections(root, manifest, args.inject_corrupt, "corrupt")

    existing = _parse_nodes(args.node)
    if existing:
        clients = {node_id: NodeClient(host, port) for node_id, (host, port) in existing.items()}
        ports = {node_id: port for node_id, (host, port) in existing.items()}
        result = run_pipeline(
            root=root,
            manifest_path=args.manifest,
            ledger_path=args.ledger,
            clients=clients,
            budget=budget,
            verify_replay=not args.no_replay,
            after_progress=_crash_hook(args.crash_after_task),
        )
        return _run_summary(result, upgraded.to_dict(), ports)
    node_ids = [node.node_id for node in manifest.nodes]
    with start_servers(root, Path(args.manifest), node_ids) as (clients, servers):
        result = run_pipeline(
            root=root,
            manifest_path=args.manifest,
            ledger_path=args.ledger,
            clients=clients,
            budget=budget,
            verify_replay=not args.no_replay,
            after_progress=_crash_hook(args.crash_after_task),
        )
        ports = {node_id: server.port for node_id, server in servers.items()}
    return _run_summary(result, upgraded.to_dict(), ports)


def _command_replay(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    upgraded = ManifestStore(root).load_current(args.manifest)
    node_ids = [node.node_id for node in upgraded.manifest.nodes]
    with start_servers(root, Path(args.manifest), node_ids) as (clients, servers):
        with Ledger(args.ledger) as ledger:
            ledger.bind_source(root, Path(args.manifest))
            recovery = ledger.open_for_state(
                upgraded.manifest,
                compute_state_digest(root, upgraded.manifest),
            )
            result = ReplayVerifier(ledger, root, clients).verify()
            return {
                **result.to_dict(),
                "recovery": recovery.to_dict(),
                "ports": {node_id: server.port for node_id, server in servers.items()},
            }


def _command_evolve(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.manifest)
    manifest = Manifest.from_file(path)
    changes: dict[str, int] = {}
    for item in args.set_priority:
        if "=" not in item:
            raise StateError("--set-priority must be NODE=PRIORITY", details={"value": item})
        node_id, raw = item.split("=", 1)
        try:
            value = int(raw)
        except ValueError as exc:
            raise StateError("priority must be an integer", details={"value": raw}) from exc
        if value <= 0:
            raise BudgetError("priority must be positive", path="$.nodes[].priority")
        changes[node_id] = value
    from dataclasses import replace

    nodes = [
        replace(node, priority=changes[node.node_id]) if node.node_id in changes else node
        for node in manifest.nodes
    ]
    result = ManifestRepository(path).save_current(
        replace(manifest, nodes=tuple(nodes))
    )
    return {"ok": True, **result.to_dict()}


def _command_inject(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    node_id, block_id = _parse_target(args.target)
    path = root / "nodes" / node_id / "blocks" / f"{block_id}.bin"
    data = path.read_bytes()
    if args.kind == "truncate":
        new_data = data[: max(0, len(data) - 1)]
        result = {"kind": "truncate", "original_size": len(data), "new_size": len(new_data)}
    else:
        new_data = bytes([data[0] ^ 0xFF]) + data[1:] if data else b"\x00"
        result = {"kind": "corrupt", "original_size": len(data), "new_size": len(new_data)}
    atomic_write_bytes(path, new_data)
    return {"ok": True, "node_id": node_id, "block_id": block_id, "path": str(path), **result}


def _apply_injections(root: Path, manifest: Manifest, values: list[str], kind: str) -> None:
    for value in values:
        node_id, block_id = _parse_target(value)
        path = root / "nodes" / node_id / "blocks" / f"{block_id}.bin"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise StateError(
                "cannot inject anomaly into missing block",
                details={"node_id": node_id, "block_id": block_id},
            ) from exc
        if kind == "truncate":
            path_data = data[: max(0, len(data) - 1)]
        else:
            path_data = bytes([data[0] ^ 0xFF]) + data[1:] if data else b"\x00"
        atomic_write_bytes(path, path_data)


def _parse_target(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise StateError("target must be NODE:BLOCK", details={"value": value})
    node_id, block_id = value.split(":", 1)
    if not node_id or not block_id:
        raise StateError("target must be NODE:BLOCK", details={"value": value})
    return node_id, block_id


def _parse_nodes(values: list[str]) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise StateError("--node must be NODE=HOST:PORT", details={"value": value})
        node_id, address = value.split("=", 1)
        host, raw_port = address.rsplit(":", 1)
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise StateError("node port must be an integer", details={"value": value}) from exc
        result[node_id] = (host, port)
    return result


def _run_summary(result, upgrade: dict[str, Any], ports: dict[str, int]) -> dict[str, Any]:
    payload = result.to_dict()
    payload["upgrade"] = upgrade
    payload["ports"] = ports
    return payload


def _crash_hook(limit: int):
    if limit <= 0:
        return None

    def hook(count: int) -> None:
        if count >= limit:
            os._exit(86)

    return hook


def _print_json(value: Any, *, stream=None) -> None:
    target = stream or sys.stdout
    target.write(canonical_json_dumps(value, pretty=True, newline=True))
    target.flush()

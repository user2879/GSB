"""Deterministic local samples for v1/v2/v3 interoperability and repair demos."""

from __future__ import annotations

import contextlib
import importlib
import platform
import socket
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from .errors import StateError
from .manifest_repo import ManifestRepository
from .manifest_store import ManifestStore
from .model import (
    ChecksumDerivation,
    BlockRecord,
    Manifest,
    NodeRecord,
    RELATION_ALGORITHM,
    derive_relation_digest,
)
from .node_service import NodeClient, NodeServer
from .planner import RepairBudget
from .util import atomic_write_bytes, atomic_write_json, sha256_bytes


SAMPLE_VERSION = 1
SAMPLE_MARKER = ".replica_repair_sample.json"


BLOCK_SIZES = {
    "blk-0001": 4096,
    "blk-0002": 2048,
    "blk-0003": 3072,
}


def deterministic_block_bytes(group_id: str, block_id: str, variant: str, size: int) -> bytes:
    seed = sha256_bytes(f"{group_id}|{block_id}|{variant}".encode("utf-8")).encode("ascii")
    repeats = (size + len(seed) - 1) // len(seed)
    return (seed * repeats)[:size]


def allocate_loopback_port(start: int = 49152) -> int:
    for port in range(start, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    if port < 1024:
        raise StateError("loopback allocator returned a non-high port", details={"port": port})
    return port


def group_specs() -> list[dict[str, Any]]:
    return [
        {
            "group_id": "interop-v1",
            "format_version": 1,
            "nodes": [
                {"node_id": "v1-a", "priority": 1, "drift": {}},
                {"node_id": "v1-b", "priority": 1, "drift": {"blk-0003": "drift-b3"}},
                {"node_id": "v1-c", "priority": 1, "drift": {"blk-0002": "drift-b2"}},
            ],
        },
        {
            "group_id": "interop-v2",
            "format_version": 2,
            "nodes": [
                {"node_id": "v2-a", "priority": 5, "drift": {}},
                {"node_id": "v2-b", "priority": 3, "drift": {}},
                {"node_id": "v2-c", "priority": 1, "drift": {"blk-0002": "drift-b2"}},
            ],
        },
        {
            "group_id": "interop-v3",
            "format_version": 3,
            "nodes": [
                {"node_id": "v3-a", "priority": 5, "drift": {}},
                {"node_id": "v3-b", "priority": 3, "drift": {"blk-0003": "drift-b3"}},
                {"node_id": "v3-c", "priority": 2, "drift": {"blk-0002": "drift-b2"}},
                {"node_id": "v3-d", "priority": 1, "drift": {"blk-0002": "drift-b2", "blk-0003": "drift-b3"}},
            ],
        },
    ]


def sample_manifest_path(root: Path, group_id: str) -> Path:
    return root / "data" / "groups" / f"{group_id}.json"


def sample_ledger_path(root: Path, group_id: str) -> Path:
    return root / "ledger" / f"{group_id}.sqlite3"


def init_sample(root: Path | str, *, force: bool = False) -> dict[str, Any]:
    root = Path(root)
    marker = root / "data" / SAMPLE_MARKER
    if marker.exists() and not force:
        raise StateError(
            "deterministic sample already exists; pass force=True or use demo --reset",
            details={"marker": str(marker)},
        )
    for directory in (root / "data", root / "data" / "groups", root / "nodes", root / "ledger"):
        directory.mkdir(parents=True, exist_ok=True)

    specs = group_specs()
    for spec in specs:
        _clean_group(root, spec["group_id"], [node["node_id"] for node in spec["nodes"]])
        _write_group(root, spec)
    marker_payload = {
        "sample_version": SAMPLE_VERSION,
        "groups": [spec["group_id"] for spec in specs],
        "block_sizes": dict(sorted(BLOCK_SIZES.items())),
        "deterministic": True,
    }
    atomic_write_json(marker, marker_payload)
    return {
        "root": str(root),
        "groups": [sample_manifest_path(root, spec["group_id"]).as_posix() for spec in specs],
        "ledgers": [sample_ledger_path(root, spec["group_id"]).as_posix() for spec in specs],
        "marker": marker_payload,
    }


def _clean_group(root: Path, group_id: str, node_ids: list[str]) -> None:
    manifest_path = sample_manifest_path(root, group_id)
    for suffix in ("", ".v1.readonly.bak", ".v2.readonly.bak", ".upgrade.checkpoint.json"):
        (Path(str(manifest_path) + suffix)).unlink(missing_ok=True)
    for node_id in node_ids:
        node_dir = root / "nodes" / node_id
        if node_dir.exists():
            for path in sorted(node_dir.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
            node_dir.rmdir()
    ledger_path = sample_ledger_path(root, group_id)
    for suffix in ("", "-wal", "-shm"):
        Path(str(ledger_path) + suffix).unlink(missing_ok=True)


def _write_group(root: Path, spec: dict[str, Any]) -> None:
    group_id = spec["group_id"]
    format_version = spec["format_version"]
    nodes: list[NodeRecord] = []
    for node_spec in spec["nodes"]:
        node_id = node_spec["node_id"]
        drift = node_spec.get("drift", {})
        blocks: list[BlockRecord] = []
        for block_id, size in sorted(BLOCK_SIZES.items()):
            variant = str(drift.get(block_id, "truth"))
            data = deterministic_block_bytes(group_id, block_id, variant, size)
            atomic_write_bytes(root / "nodes" / node_id / "blocks" / f"{block_id}.bin", data)
            digest = sha256_bytes(data) if format_version >= 2 else None
            blocks.append(
                BlockRecord(
                    block_id=block_id,
                    size_bytes=size,
                    digest=digest,
                    repair_weight={"blk-0001": 5, "blk-0002": 9, "blk-0003": 5}[block_id],
                    checksum_derivation=None,
                    repair_source="origin",
                )
            )
        if format_version == 3:
            by_id = {block.block_id: block for block in blocks}
            relation = ChecksumDerivation(
                algorithm=RELATION_ALGORITHM,
                source_block_ids=("blk-0001", "blk-0002"),
                digest=derive_relation_digest(
                    RELATION_ALGORITHM,
                    ("blk-0001", "blk-0002"),
                    {
                        "blk-0001": by_id["blk-0001"].digest,
                        "blk-0002": by_id["blk-0002"].digest,
                    },
                ),
            )
            blocks = [
                replace(block, checksum_derivation=relation) if block.block_id == "blk-0003" else block
                for block in blocks
            ]
        nodes.append(
            NodeRecord(
                node_id=node_id,
                priority=node_spec["priority"] if format_version >= 2 else None,
                blocks=tuple(blocks),
            )
        )
    manifest = Manifest(
        format_version=format_version,
        group_id=group_id,
        manifest_version=1,
        nodes=tuple(nodes),
    )
    payload = manifest.to_dict(format_version)
    if format_version == 3:
        Manifest.from_dict(payload)
    path = sample_manifest_path(root, group_id)
    import json

    atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )


@contextlib.contextmanager
def start_servers(
    root: Path,
    manifest_path: Path,
    node_ids: list[str],
) -> Iterator[tuple[dict[str, NodeClient], dict[str, NodeServer]]]:
    clients: dict[str, NodeClient] = {}
    servers: dict[str, NodeServer] = {}
    try:
        for node_id in node_ids:
            server = NodeServer(root, manifest_path, node_id, port=0).start()
            servers[node_id] = server
            clients[node_id] = NodeClient(server.host, server.port)
        yield clients, servers
    finally:
        for server in servers.values():
            server.stop()


def demo_group(root: Path, group_id: str, *, budget: RepairBudget) -> dict[str, Any]:
    manifest_path = sample_manifest_path(root, group_id)
    ledger_path = sample_ledger_path(root, group_id)
    upgrade = ManifestStore(root).load_current(manifest_path)
    manifest = upgrade.manifest
    node_ids = [node.node_id for node in manifest.nodes]
    with start_servers(root, manifest_path, node_ids) as (clients, servers):
        from .engine import run_pipeline

        pipeline = run_pipeline(
            root=root,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            clients=clients,
            budget=budget,
            verify_replay=True,
        )
        ports = {node_id: server.port for node_id, server in servers.items()}
    report = pipeline.ledger_report
    current = report["current"]
    baseline = report["baseline"]
    return {
        "group_id": group_id,
        "upgrade": upgrade.to_dict(),
        "ports": ports,
        "manifest_version": pipeline.manifest.manifest_version,
        "before": {
            "snapshot_id": baseline["snapshot_id"] if baseline else pipeline.before_snapshot_id,
            "mismatches": baseline["mismatches"] if baseline else [],
            "truth": baseline["truth"] if baseline else [],
            "repair_stats": baseline["repair_stats"] if baseline else {},
        },
        "plan": pipeline.plan.to_dict(),
        "execution": pipeline.execution.to_dict() if pipeline.execution else None,
        "after": {
            "snapshot_id": current["snapshot_id"],
            "mismatches": current["mismatches"],
            "truth": current["truth"],
            "repair_stats": current["repair_stats"],
        },
        "recovery": pipeline.recovery,
        "reconciliation": pipeline.reconciliation,
        "replay": pipeline.replay,
        "ledger": {
            "generation": report["generation"],
            "content_digest": report["content_digest"],
            "stale": report["stale"],
            "progress": report["progress"],
        },
    }


def evolve_priority(root: Path, group_id: str, node_id: str, priority: int) -> dict[str, Any]:
    path = sample_manifest_path(root, group_id)
    manifest = Manifest.from_file(path)
    nodes = []
    found = False
    for node in manifest.nodes:
        if node.node_id == node_id:
            nodes.append(replace(node, priority=priority))
            found = True
        else:
            nodes.append(node)
    if not found:
        raise StateError("sample node is absent", details={"node_id": node_id})
    candidate = replace(manifest, nodes=tuple(nodes))
    result = ManifestRepository(path).save_current(candidate)
    return result.to_dict()


def environment_report() -> dict[str, Any]:
    numpy_version = None
    try:
        numpy_version = importlib.import_module("numpy").__version__
    except Exception as exc:
        numpy_version = f"unavailable:{type(exc).__name__}"
    tkinter_version = None
    try:
        tkinter_version = str(importlib.import_module("tkinter").TkVersion)
    except Exception as exc:
        tkinter_version = f"unavailable:{type(exc).__name__}"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "numpy": numpy_version,
        "tkinter": tkinter_version,
        "loopback_probe_port": allocate_loopback_port(),
    }


def run_demo(root: Path | str, *, force_reset: bool = True) -> dict[str, Any]:
    root = Path(root)
    init = init_sample(root, force=force_reset)
    budgets = {
        "interop-v1": RepairBudget(max_parallel=3, per_node_kb=3),
        "interop-v2": RepairBudget(max_parallel=3, per_node_kb=3),
        "interop-v3": RepairBudget(max_parallel=3, per_node_kb=3),
    }
    groups = [demo_group(root, group_id, budget=budgets[group_id]) for group_id in (
        "interop-v1",
        "interop-v2",
        "interop-v3",
    )]
    evolution = evolve_priority(root, "interop-v3", "v3-c", 6)
    v3_after_evolution = demo_group(
        root,
        "interop-v3",
        budget=RepairBudget(max_parallel=3, per_node_kb=3),
    )
    summary = {
        "ok": True,
        "sample": init,
        "environment": environment_report(),
        "groups": groups,
        "stale_protection": {
            "evolution": evolution,
            "run": {
                "recovery": v3_after_evolution["recovery"],
                "manifest_version": v3_after_evolution["manifest_version"],
                "ledger": v3_after_evolution["ledger"],
                "replay": v3_after_evolution["replay"],
            },
        },
    }
    atomic_write_json(root / "data" / "demo_report.json", summary)
    return summary

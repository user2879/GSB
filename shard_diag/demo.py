from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


def free_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_demo_workspace(root: str | Path, *, global_budget_units: int = 10) -> dict[str, Any]:
    root = Path(root)
    (root / "declarations").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "ledger" / "logs").mkdir(parents=True, exist_ok=True)
    ports = [free_loopback_port() for _ in range(3)]
    nodes_doc = {
        "format_version": 1,
        "nodes": [
            {
                "node_id": "node-a", "host": "127.0.0.1", "port": ports[0], "seed": 1701,
                "fixed_scale": 100, "weights_q": [100, -200, 150, 50], "bias_q": 25,
                "concurrency": 2, "lease_limit": 2, "startup_timeout_ms": 5000,
                "socket_timeout_ms": 1000,
            },
            {
                "node_id": "node-b", "host": "127.0.0.1", "port": ports[1], "seed": 2702,
                "fixed_scale": 1000, "weights_q": [900, -2100, 1600, -400], "bias_q": 120,
                "concurrency": 2, "lease_limit": 2, "startup_timeout_ms": 5000,
                "socket_timeout_ms": 1000,
            },
            {
                "node_id": "node-c", "host": "127.0.0.1", "port": ports[2], "seed": 3703,
                "fixed_scale": 1024, "weights_q": [1024, -2048, 1024, 0], "bias_q": 16,
                "concurrency": 2, "lease_limit": 2, "startup_timeout_ms": 5000,
                "socket_timeout_ms": 1000,
            },
        ],
    }
    shards_doc = {
        "format_version": 1,
        "shards": [
            {"shard_id": "shard-a", "node_id": "node-a", "upstream": [], "sample_rows": 12, "resource_units": 2, "priority": 10, "batch_count_hint": 3},
            {"shard_id": "shard-b", "node_id": "node-b", "upstream": [], "sample_rows": 10, "resource_units": 2, "priority": 0, "batch_count_hint": 2},
            {"shard_id": "shard-c", "node_id": "node-c", "upstream": ["shard-a", "shard-b"], "sample_rows": 8, "resource_units": 2, "priority": 0, "batch_count_hint": 2},
            {"shard_id": "shard-d", "node_id": "node-a", "upstream": ["shard-a"], "sample_rows": 6, "resource_units": 2, "priority": 0, "batch_count_hint": 2},
            {"shard_id": "shard-e", "node_id": "node-b", "upstream": ["shard-b"], "sample_rows": 6, "resource_units": 2, "priority": 0, "batch_count_hint": 2},
            {"shard_id": "shard-f", "node_id": "node-c", "upstream": ["shard-c", "shard-d", "shard-e"], "sample_rows": 9, "resource_units": 1, "priority": 0, "batch_count_hint": 3},
        ],
    }
    quota_doc = {
        "format_version": 1,
        "versions": [
            {
                "version": 1, "parent_version": None, "session_scale": 1000,
                "global_budget_units": global_budget_units,
                "upsert_node_quotas": {
                    "node-a": {"resource_units_cap": 6, "concurrency": 2, "lease_limit": 2},
                    "node-b": {"resource_units_cap": 6, "concurrency": 2, "lease_limit": 2},
                    "node-c": {"resource_units_cap": 6, "concurrency": 2, "lease_limit": 2},
                },
                "delete_node_quotas": [],
                "upsert_batch_requirements": {"default": 1},
                "delete_batch_requirements": [],
            }
        ],
    }
    events_doc = {
        "format_version": 1,
        "events": [{"seq": 1, "type": "workspace_initialized", "at": "2026-11-27T00:00:00+08:00", "payload": {"demo": True}}],
    }
    overrides_doc = {"format_version": 1, "overrides": {}}
    write_json(root / "declarations" / "nodes.json", nodes_doc)
    write_json(root / "declarations" / "shards.json", shards_doc)
    write_json(root / "declarations" / "quota_versions.json", quota_doc)
    write_json(root / "declarations" / "events.json", events_doc)
    write_json(root / "samples" / "overrides.json", overrides_doc)
    return {"root": str(root), "ports": ports, "nodes_doc": nodes_doc, "shards_doc": shards_doc}

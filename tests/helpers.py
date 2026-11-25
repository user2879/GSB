from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def manifest_data(
    ports: list[int],
    *,
    ttl_ms: int = 5000,
    sample_scale: int = 64,
    shard_count: int = 4,
    events: list[dict[str, Any]] | None = None,
    node_count: int = 2,
    slots: int = 2,
    lease_limit: int = 2,
) -> dict[str, Any]:
    nodes = []
    for index in range(node_count):
        nodes.append(
            {
                "node_id": f"n{index + 1}",
                "host": "127.0.0.1",
                "port": ports[index],
                "slots": slots,
                "lease_limit": lease_limit,
                "seed": 1000 + index,
                "quota_version": 1,
            }
        )
    return {
        "format_version": 1,
        "config": {
            "sample_scale": sample_scale,
            "lease_ttl_ms": ttl_ms,
            "global_lease_limit": 8,
            "global_byte_budget": 131072,
            "socket_timeout_ms": 2000,
            "handshake_timeout_ms": 3000,
            "max_workers": 8,
            "drain_timeout_ms": 2000,
        },
        "nodes": nodes,
        "contents": [
            {
                "content_key": "clip",
                "total_bytes": 4096,
                "shard_count": shard_count,
                "content_seed": 7001,
            }
        ],
        "events": events or [],
    }


def write_manifest(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

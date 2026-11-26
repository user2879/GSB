from __future__ import annotations

import copy
import socket
from typing import Any


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


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


def base_raw_spec(*, global_budget: int = 6, max_workers: int = 3) -> dict[str, Any]:
    ports = free_ports(4)
    return {
        "format_version": 1,
        "capacity_version": "cap-test",
        "fixed_scale": 1000,
        "global_budget": global_budget,
        "max_workers": max_workers,
        "lease_ttl_ms": 2500,
        "handshake_timeout_ms": 1200,
        "io_timeout_ms": 1200,
        "poll_interval_ms": 2,
        "nodes": [
            {
                "id": "node-a",
                "host": "127.0.0.1",
                "port": ports[0],
                "seed": 1101,
                "concurrency": 2,
                "lease_limit": 2,
                "quota": 10,
                "used": 2,
                "fixed_scale": 1000,
            },
            {
                "id": "node-b",
                "host": "127.0.0.1",
                "port": ports[1],
                "seed": 2202,
                "concurrency": 2,
                "lease_limit": 2,
                "quota": 8,
                "used": 1,
                "fixed_scale": 1000,
            },
            {
                "id": "node-c",
                "host": "127.0.0.1",
                "port": ports[2],
                "seed": 3303,
                "concurrency": 2,
                "lease_limit": 2,
                "quota": 6,
                "used": 0,
                "fixed_scale": 10000,
            },
            {
                "id": "node-d",
                "host": "127.0.0.1",
                "port": ports[3],
                "seed": 4404,
                "concurrency": 2,
                "lease_limit": 2,
                "quota": 7,
                "used": 1,
                "fixed_scale": 1000,
            },
        ],
        "shards": [
            {
                "id": "raw-a",
                "node_id": "node-a",
                "dependencies": [],
                "demand": 3,
                "sample_count": 24,
                "group_count": 4,
                "batch_size": 7,
            },
            {
                "id": "raw-b",
                "node_id": "node-b",
                "dependencies": [],
                "demand": 3,
                "sample_count": 20,
                "group_count": 5,
                "batch_size": 6,
            },
            {
                "id": "raw-d",
                "node_id": "node-d",
                "dependencies": [],
                "demand": 2,
                "sample_count": 15,
                "group_count": 3,
                "batch_size": 5,
            },
            {
                "id": "join-ab",
                "node_id": "node-c",
                "dependencies": ["raw-a", "raw-b"],
                "demand": 2,
                "sample_count": 18,
                "group_count": 3,
                "batch_size": 4,
            },
            {
                "id": "report",
                "node_id": "node-d",
                "dependencies": ["join-ab", "raw-d"],
                "demand": 2,
                "sample_count": 12,
                "group_count": 2,
                "batch_size": 3,
            },
        ],
    }


def evolved_quota_spec(raw: dict[str, Any], *, node_index: int = 0, quota: int = 9) -> dict[str, Any]:
    evolved = copy.deepcopy(raw)
    evolved["nodes"][node_index]["quota"] = quota
    return evolved

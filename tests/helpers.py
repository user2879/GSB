from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from typing import Any


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            if port < 1024:
                sock.close()
                continue
            sockets.append(sock)
            ports.append(port)
        if len(ports) < count:
            raise RuntimeError("could not allocate enough high loopback ports")
        return ports
    finally:
        for sock in sockets:
            sock.close()


def make_spec(
    ports: list[int],
    *,
    stages: list[dict[str, Any]] | None = None,
    global_concurrency: int = 3,
    lease_ms: int = 150,
    seeds: list[int] | None = None,
    capacities: list[int] | None = None,
    limits: list[int] | None = None,
    node_ids: list[str] | None = None,
) -> dict[str, Any]:
    node_ids = node_ids or [f"node-{chr(97 + index)}" for index in range(len(ports))]
    seeds = seeds or [101 + index * 101 for index in range(len(ports))]
    capacities = capacities or [4] * len(ports)
    limits = limits or [2] * len(ports)
    stages = stages or [
        {
            "id": "build",
            "concurrency": 2,
            "artifacts": [
                {"key": "src/a", "references": []},
                {"key": "src/b", "references": []},
                {"key": "pkg/all", "references": ["src/a", "src/b"]},
            ],
        }
    ]
    return {
        "format_version": 1,
        "nodes": [
            {
                "id": node_id,
                "host": "127.0.0.1",
                "port": port,
                "capacity_slots": capacities[index],
                "lease_limit": limits[index],
                "random_seed": seeds[index],
            }
            for index, (node_id, port) in enumerate(zip(node_ids, ports))
        ],
        "pipeline": {
            "id": "test-pipeline",
            "global_concurrency": global_concurrency,
            "lease_ms": lease_ms,
            "stages": stages,
        },
    }


def write_spec(directory: Path, raw: dict[str, Any], name: str = "spec.json") -> Path:
    path = directory / name
    path.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_tempdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="leasecache-test-"))

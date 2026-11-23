from __future__ import annotations

import json
import socket
from pathlib import Path


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def manifest(
    *,
    ports: tuple[int, ...] | None = None,
    budget: int = 3,
    task_shards: int = 3,
    task_limit: int = 2,
    cluster_quotas: tuple[int, ...] | None = None,
    resource_units: int = 1,
    config_keys: tuple[str, ...] = ("scaled",),
    targets: tuple[str, ...] | None = None,
) -> dict:
    if ports is None:
        ports = (free_port(),)
    cluster_quotas = cluster_quotas or tuple(2 for _ in ports)
    cluster_ids = [f"c{index}" for index in range(len(ports))]
    targets = targets or tuple(cluster_ids)
    return {
        "global_resource_budget": budget,
        "clusters": [
            {
                "id": cluster_id,
                "host": "127.0.0.1",
                "port": port,
                "nodes": [f"{cluster_id}-n1"],
                "concurrency": cluster_quotas[index],
                "seed": 100 + index,
            }
            for index, (cluster_id, port) in enumerate(zip(cluster_ids, ports))
        ],
        "tasks": [
            {
                "id": "task",
                "config_keys": list(config_keys),
                "target_clusters": list(targets),
                "shards": task_shards,
                "resource_units": resource_units,
                "concurrency_limit": task_limit,
            }
        ],
        "revisions": [
            {
                "version": 1,
                "parent": None,
                "values": {"base": 10, "unused": 1},
                "derived": {"scaled": {"op": "sum", "inputs": ["base"]}},
            }
        ],
    }


def evolved(base: dict, *, version: int = 2, values: dict | None = None) -> dict:
    result = json.loads(json.dumps(base))
    result["revisions"].append(
        {
            "version": version,
            "parent": version - 1,
            "values": dict(values or {"base": 12}),
        }
    )
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")

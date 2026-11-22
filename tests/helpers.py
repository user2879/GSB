from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def behavior(
    *,
    fast_ms: int = 4,
    slow_ms: int = 40,
    success: int = 100,
    slow: int = 0,
    error: int = 0,
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "fast_ms": fast_ms,
        "slow_ms": slow_ms,
        "success_percent": success,
        "slow_percent": slow,
        "error_percent": error,
        "rules": rules or [],
    }


def make_spec(
    services: list[dict[str, Any]],
    *,
    ports: list[int] | None = None,
    global_concurrency: int = 3,
    timeout_ms: int = 150,
    backoffs: list[int] | None = None,
    failure_threshold: int = 2,
    cooldown_ms: int = 150,
    half_open_probes: int = 1,
    scale: int = 1_000_000,
) -> dict[str, Any]:
    if ports is not None:
        cursor = 0
        for service in services:
            for instance in service["instances"]:
                instance.setdefault("host", "127.0.0.1")
                instance.setdefault("port", ports[cursor])
                cursor += 1
    rules = []
    for service in services:
        rules.append(
            {
                "service": service["id"],
                "weights": {
                    instance["id"]: instance.get("weight", 1)
                    for instance in service["instances"]
                },
            }
        )
    return {
        "schema_version": 1,
        "session": {
            "scale": scale,
            "global_concurrency": global_concurrency,
            "attempt_timeout_ms": timeout_ms,
            "retry_backoff_ms": [4, 12] if backoffs is None else backoffs,
            "breaker": {
                "failure_threshold": failure_threshold,
                "cooldown_ms": cooldown_ms,
                "half_open_probes": half_open_probes,
            },
        },
        "services": services,
        "route_revisions": [
            {"revision": 1, "description": "test", "rules": rules}
        ],
    }


def instance(
    instance_id: str,
    *,
    weight: int = 1,
    seed: int = 7,
    host: str = "127.0.0.1",
    port: int | None = None,
    behavior_value: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "id": instance_id,
        "host": host,
        "port": port if port is not None else 1,
        "weight": weight,
        "seed": seed,
        "behavior": behavior_value or behavior(),
    }
    return result


def write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def requests_payload(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {"requests": requests}

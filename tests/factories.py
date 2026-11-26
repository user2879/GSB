from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_spec(path: Path, raw: dict[str, Any]) -> Path:
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sample_spec_with_free_ports() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    raw = json.loads((root / "samples" / "local_spec.json").read_text(encoding="utf-8"))
    for node in raw["nodes"]:
        node["port"] = free_tcp_port()
        node["processing_delay_ms"] = 5
    raw["scheduling"]["handshake_timeout_ms"] = 10_000
    raw["scheduling"]["request_timeout_ms"] = 10_000
    raw["scheduling"]["lease_ttl_ms"] = 10_000
    return raw


def single_node_spec(tmp_path: Path, **overrides: Any) -> tuple[Path, dict[str, Any]]:
    raw: dict[str, Any] = {
        "schema_version": 1,
        "session_id": "integration",
        "scale_factor": 100,
        "budget": {"resource_units": overrides.pop("budget_units", 100)},
        "scheduling": {
            "lease_ttl_ms": 10_000,
            "request_timeout_ms": 10_000,
            "handshake_timeout_ms": 10_000,
            "poll_interval_ms": 5,
        },
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": free_tcp_port(),
                "concurrency": overrides.pop("concurrency", 2),
                "lease_limit": overrides.pop("lease_limit", 2),
                "seed": 4242,
                "capacity_version": 1,
                "fixed_scale": 100,
                "processing_delay_ms": overrides.pop("processing_delay_ms", 20),
            }
        ],
        "shards": overrides.pop(
            "shards",
            [
                {
                    "shard_id": "s1",
                    "node_id": "n1",
                    "dependencies": [],
                    "rows": 16,
                    "resource_units": 1,
                    "priority": 10,
                },
                {
                    "shard_id": "s2",
                    "node_id": "n1",
                    "dependencies": [],
                    "rows": 16,
                    "resource_units": 1,
                    "priority": 9,
                },
            ],
        ),
    }
    raw.update(overrides)
    spec_path = write_spec(tmp_path / "spec.json", raw)
    return spec_path, raw

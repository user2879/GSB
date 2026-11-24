from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from cachelease.canonical import canonical_json


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def one_node_config(
    tmp_path: Path,
    *,
    capacity_slots: int = 4,
    lease_limit: int = 4,
    artifacts: list[dict[str, Any]] | None = None,
    grace_ms: int = 200,
    request_concurrency: int = 2,
    seed: int = 17,
) -> tuple[dict[str, Any], Path]:
    port = free_port()
    if artifacts is None:
        artifacts = [
            {"key": "core", "stage": "build", "refs": [], "shards": {"count": 2, "bytes": 32}},
            {"key": "api", "stage": "build", "refs": ["core"], "shards": {"count": 2, "bytes": 32}},
        ]
    config = {
        "format_version": 1,
        "session": {"scale": 1000},
        "runtime": {
            "global_concurrency": 4,
            "handshake_timeout_ms": 3000,
            "reply_timeout_ms": 3000,
            "grace_ms": grace_ms,
        },
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": port,
                "capacity_slots": capacity_slots,
                "lease_limit": lease_limit,
                "seed": seed,
                "quota_version": 1,
                "request_concurrency": request_concurrency,
            }
        ],
        "pipeline": {
            "stages": [{"stage": "build", "concurrency": 2}],
            "artifacts": artifacts,
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config, path


def two_node_config(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    ports = [free_port(), free_port()]
    config = {
        "format_version": 1,
        "session": {"scale": 1000},
        "runtime": {
            "global_concurrency": 2,
            "handshake_timeout_ms": 3000,
            "reply_timeout_ms": 3000,
            "grace_ms": 200,
        },
        "nodes": [
            {
                "node_id": node_id,
                "host": "127.0.0.1",
                "port": port,
                "capacity_slots": 3,
                "lease_limit": 3,
                "seed": 10 + index,
                "quota_version": 1,
                "request_concurrency": 1,
            }
            for index, (node_id, port) in enumerate(zip(("node-a", "node-b"), ports))
        ],
        "pipeline": {
            "stages": [{"stage": "build", "concurrency": 2}],
            "artifacts": [
                {"key": "core", "stage": "build", "refs": [], "shards": {"count": 1, "bytes": 16}}
            ],
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config-two.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config, path


def write_events(path: Path, events: list[dict[str, Any]]) -> Path:
    path.write_text("".join(canonical_json(event) + "\n" for event in events), encoding="utf-8")
    return path


def write_sample(tmp_path: Path, seed: int = 777) -> tuple[Path, Path, int, dict[str, Any]]:
    from cachelease.samplegen import build_sample

    config, events, checkpoint = build_sample(seed)
    config_path = tmp_path / "sample.config.json"
    events_path = tmp_path / "events.ndjson"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_events(events_path, events)
    return config_path, events_path, checkpoint, config

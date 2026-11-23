from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def default_subscribers(count: int = 1, *, lease_limit: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "subscriber_id": f"s{index}",
            "host": "127.0.0.1",
            "port": free_port(),
            "processing_budget_ms": 120,
            "lease_limit": lease_limit,
            "seed": 100 + index,
        }
        for index in range(count)
    ]


def default_channels(*, batch_size: int = 1, rate: int = 50, quota: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "channel_id": "ch-orders",
            "topic": "orders",
            "batch_size": batch_size,
            "send_rate_limit": rate,
            "subscriber_quota": quota,
        }
    ]


def write_spec(
    path: Path,
    *,
    version: int = 1,
    global_lease_limit: int = 3,
    handshake_timeout_ms: int = 3000,
    subscribers: list[dict[str, Any]] | None = None,
    channels: list[dict[str, Any]] | None = None,
) -> Path:
    payload = {
        "declaration_version": version,
        "global_lease_limit": global_lease_limit,
        "handshake_timeout_ms": handshake_timeout_ms,
        "subscribers": subscribers or default_subscribers(),
        "channels": channels or default_channels(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_events(path: Path, count: int, *, topic: str = "orders") -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(1, count + 1):
            handle.write(
                json.dumps(
                    {
                        "seq": index,
                        "ts_ms": 1000 + index // 2,
                        "topic": topic,
                        "payload_digest": f"sha256:test-{index}",
                        "expected_order": index,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return path


def write_topic_events(path: Path, topics: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, topic in enumerate(topics, start=1):
            handle.write(
                json.dumps(
                    {
                        "seq": index,
                        "ts_ms": 1000 + index,
                        "topic": topic,
                        "payload_digest": f"sha256:topic-{index}",
                        "expected_order": index,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return path

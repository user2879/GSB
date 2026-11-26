"""回环 JSON 行协议常量与帧编解码（主/子进程共用）。"""
from __future__ import annotations

import json
from typing import Any

LINE_TERMINATOR = b"\n"
HANDSHAKE_REQUEST = "DT_HANDSHAKE"
HANDSHAKE_ACK = "DT_READY"
DISPATCH = "DT_DISPATCH"
SHUTDOWN = "DT_SHUTDOWN"
LEASE_CHECK = "DT_LEASE_CHECK"

REQUIRED_RECEIPT_FIELDS = (
    "node_id", "shard_id", "param_version", "summary", "elapsed_ms",
)


def encode_frame(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False)
            + LINE_TERMINATOR.decode("utf-8")).encode("utf-8")


def decode_frame(line: bytes) -> dict[str, Any]:
    return json.loads(line.decode("utf-8"))

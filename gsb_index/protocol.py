"""JSON-lines loopback protocol shared by client, server and tests."""

from __future__ import annotations

import json
import socket
from typing import Any

from .canonical import canonical_json


MAX_LINE_BYTES = 16 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    return (canonical_json(message) + "\n").encode("utf-8")


def decode_message(line: bytes) -> dict[str, Any]:
    if not line.endswith(b"\n"):
        line += b"\n"
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol message must be a JSON object")
    return value


class ShardClient:
    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, message: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        with socket.create_connection((self.host, self.port), timeout=timeout or self.timeout) as connection:
            connection.settimeout(timeout or self.timeout)
            connection.sendall(encode_message(message))
            buffer = bytearray()
            while not buffer.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    raise ConnectionError("shard closed the connection before a response line")
                buffer.extend(chunk)
                if len(buffer) > MAX_LINE_BYTES:
                    raise ValueError("shard response exceeded MAX_LINE_BYTES")
            return decode_message(bytes(buffer))


def request_batch(
    *,
    request_id: str,
    shard_id: str,
    revision: int,
    documents: list[dict[str, Any]],
    compression: dict[str, Any],
    entry_limit: int,
    debug_delay_ms: int = 0,
) -> dict[str, Any]:
    return {
        "type": "index_batch",
        "request_id": request_id,
        "shard_id": shard_id,
        "revision": revision,
        "documents": documents,
        "compression": compression,
        "entry_limit": entry_limit,
        "debug_delay_ms": debug_delay_ms,
    }


"""Minimal newline-delimited JSON TCP client used by nodes and the demo."""

from __future__ import annotations

import json
import socket
from typing import Any


MAX_LINE_BYTES = 1_048_576


def request_json(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.sendall(data)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_LINE_BYTES:
                raise ValueError("response line exceeds protocol limit")
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        raise ValueError("empty response from coordination server")
    response = json.loads(raw.decode("utf-8"))
    if not isinstance(response, dict):
        raise ValueError("response must be a JSON object")
    return response


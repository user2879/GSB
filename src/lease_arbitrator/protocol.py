from __future__ import annotations

import json
import socket
from typing import Any

from .errors import NodeProtocolError, NodeUnavailableError


PROTOCOL_VERSION = 1


def dumps_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def read_json_line(reader: Any, *, timeout: float, stage: str) -> dict[str, Any]:
    try:
        line = reader.readline()
    except (socket.timeout, TimeoutError, OSError) as exc:
        raise NodeProtocolError(f"{stage} read timeout: {exc}", details={"stage": stage}) from exc
    if not line:
        raise NodeUnavailableError(f"{stage}: connection closed before response", details={"stage": stage})
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NodeProtocolError(f"{stage}: invalid JSON line: {exc}", details={"stage": stage}) from exc
    if not isinstance(value, dict):
        raise NodeProtocolError(f"{stage}: response must be an object", details={"stage": stage})
    return value


def request_json_line(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout: float = 2.0,
    handshake: bool = True,
) -> dict[str, Any]:
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except (OSError, socket.timeout) as exc:
        raise NodeUnavailableError(
            f"cannot connect to {host}:{port}: {exc}",
            details={"host": host, "port": port},
        ) from exc
    with sock:
        sock.settimeout(timeout)
        reader = sock.makefile("rwb", buffering=0)
        try:
            if handshake:
                hello = {"type": "hello", "protocol": PROTOCOL_VERSION, "client": "lease-arbitrator"}
                sock.sendall(dumps_line(hello))
                ack = read_json_line(reader, timeout=timeout, stage="handshake")
                if not ack.get("ok"):
                    raise NodeProtocolError("handshake rejected", details={"response": ack})
            sock.sendall(dumps_line(payload))
            response = read_json_line(reader, timeout=timeout, stage=payload.get("type", "request"))
            return response
        except (socket.timeout, TimeoutError) as exc:
            raise NodeProtocolError(
                f"socket timeout during {payload.get('type', 'request')}",
                details={"host": host, "port": port},
            ) from exc


def wait_for_port(host: str, port: int, *, timeout: float = 5.0, request_timeout: float = 0.25) -> None:
    import time

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = request_json_line(
                host,
                port,
                {"type": "ping"},
                timeout=request_timeout,
            )
            if response.get("ok"):
                return
        except Exception as exc:  # startup race: retry until deadline
            last_error = exc
        time.sleep(0.02)
    raise NodeUnavailableError(
        f"node at {host}:{port} did not become ready within {timeout}s",
        details={"last_error": str(last_error)},
    )

"""Loopback JSON-lines protocol helpers with real socket timeouts."""

from __future__ import annotations

import json
import os
import socket
from typing import Any

from .util import canonical_bytes


MAX_LINE_BYTES = 4 * 1024 * 1024
PROTOCOL_VERSION = 1


class ProtocolEOF(Exception):
    pass


class ProtocolError(Exception):
    pass


def send_message(sock: socket.socket, value: Any) -> None:
    sock.sendall(canonical_bytes(value) + b"\n")


def receive_message(sock: socket.socket) -> dict[str, Any]:
    buffer = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ProtocolEOF("peer closed before a complete JSON line")
        buffer.extend(chunk)
        if len(buffer) > MAX_LINE_BYTES:
            raise ProtocolError("JSON line exceeds 4 MiB")
        newline = buffer.find(b"\n")
        if newline >= 0:
            line = bytes(buffer[:newline])
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError(f"invalid JSON line: {exc}") from exc
            if not isinstance(value, dict):
                raise ProtocolError("protocol message must be a JSON object")
            return value


def connect(host: str, port: int, timeout_ms: int) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout_ms / 1000.0)
    sock.settimeout(timeout_ms / 1000.0)
    return sock


def is_port_free(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.25)
        if os.name == "nt":
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()

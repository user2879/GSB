"""JSON-lines-over-loopback-TCP protocol helpers."""

from __future__ import annotations

import json
import socket

from .errors import BizError, ErrorCode
from .manifest import canonical_json


def send_message(sock: socket.socket, payload: dict) -> None:
    sock.sendall((canonical_json(payload) + "\n").encode("utf-8"))


def recv_message(sock: socket.socket) -> dict:
    buffer = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise BizError(
                ErrorCode.E_PROTO,
                "node closed connection before sending a complete message",
                {"received": buffer.decode("utf-8", errors="replace")},
            )
        buffer.extend(chunk)
        if b"\n" in buffer:
            line, _, _ = buffer.partition(b"\n")
            try:
                return json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise BizError(
                    ErrorCode.E_PROTO,
                    f"invalid JSON line from node: {exc.msg}",
                    {"line": line.decode("utf-8", errors="replace")},
                ) from exc


def request(host: str, port: int, payload: dict, timeout_ms: int) -> dict:
    with socket.create_connection((host, port), timeout=timeout_ms / 1000.0) as sock:
        sock.settimeout(timeout_ms / 1000.0)
        send_message(sock, payload)
        return recv_message(sock)


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((host, port)) == 0

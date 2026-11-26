from __future__ import annotations

import json
import socket
from typing import Any

from .errors import Codes, DiagnosticError


def encode_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def send_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_line(payload))


def read_line(sock: socket.socket, *, timeout: float = 2.0) -> dict[str, Any]:
    sock.settimeout(timeout)
    chunks = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout as exc:
            raise DiagnosticError(Codes.TIMEOUT, f"socket read timed out after {timeout:.3f}s", field="socket") from exc
        except TimeoutError as exc:  # pragma: no cover - platform dependent
            raise DiagnosticError(Codes.TIMEOUT, "socket read timed out", field="socket") from exc
        if not chunk:
            raise DiagnosticError(Codes.PORT, "node closed JSON-lines connection", field="socket")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.extend(chunk[:newline])
            break
        chunks.extend(chunk)
        if len(chunks) > 16 * 1024 * 1024:
            raise DiagnosticError(Codes.FORMAT, "JSON line exceeds 16 MiB safety limit", field="socket")
    try:
        value = json.loads(chunks.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(Codes.FORMAT, f"invalid JSON line from node: {exc}", field="socket") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(Codes.FORMAT, "node response must be a JSON object", field="socket")
    if value.get("ok") is False and "error" in value:
        error = value["error"]
        if isinstance(error, dict) and error.get("code") in {Codes.LEASE, Codes.TIMEOUT, Codes.PORT}:
            raise DiagnosticError(error["code"], error.get("message", "node rejected request"), details=error)
    return value


def request_response(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout: float = 3.0,
    handshake_node_id: str | None = None,
) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            send_line(sock, {"type": "handshake", "node_id": handshake_node_id})
            hello = read_line(sock, timeout=timeout)
            if not hello.get("ok"):
                raise DiagnosticError(Codes.PORT, f"bad handshake: {hello}", field="handshake")
            send_line(sock, payload)
            return read_line(sock, timeout=timeout)
    except socket.timeout as exc:
        raise DiagnosticError(Codes.TIMEOUT, f"connect/request timed out to {host}:{port}", field="port") from exc
    except ConnectionRefusedError as exc:
        raise DiagnosticError(Codes.PORT, f"connection refused on {host}:{port}", field="port") from exc
    except OSError as exc:
        if getattr(exc, "errno", None) in {98, 10048}:
            raise DiagnosticError(Codes.PORT, f"port already in use: {host}:{port}", field="port") from exc
        raise DiagnosticError(Codes.PORT, f"socket error on {host}:{port}: {exc}", field="port") from exc


def renew_lease_request(
    host: str,
    port: int,
    *,
    node_id: str,
    lease_id: str,
    expected_version: int,
    timeout: float = 2.0,
) -> dict[str, Any]:
    return request_response(
        host,
        port,
        {
            "type": "renew",
            "node_id": node_id,
            "lease_id": lease_id,
            "lease_version": expected_version,
        },
        timeout=timeout,
        handshake_node_id=node_id,
    )

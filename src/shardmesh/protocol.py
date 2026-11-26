"""JSON-lines TCP protocol helpers."""

from __future__ import annotations

import errno
import json
import socket
from typing import Any, Iterable

from .errors import ShardMeshError

MAX_LINE_BYTES = 256 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _timeout_error(action: str, host: str, port: int) -> ShardMeshError:
    return ShardMeshError(
        "E_TIMEOUT",
        f"timed out while {action} {host}:{port}",
        category="tcp",
        field_name="socket_timeout",
        details={"host": host, "port": port},
    )


def _field_error(message: str, *, port: int | None = None, details: dict[str, Any] | None = None) -> ShardMeshError:
    return ShardMeshError(
        "E_FIELD",
        message,
        category="protocol",
        field_name="response",
        details={"port": port, **(details or {})},
    )


def assert_port_available(host: str, port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise ShardMeshError(
            "E_PORT",
            f"loopback port {port} is unavailable: {exc}",
            category="node",
            field_name="port",
            details={"host": host, "port": port, "errno": exc.errno},
        ) from exc
    finally:
        probe.close()


def read_json_line(sock: socket.socket, max_bytes: int = MAX_LINE_BYTES) -> dict[str, Any]:
    chunks = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            if not chunks:
                raise ConnectionResetError("peer closed connection without a JSON line")
            break
        chunks.extend(chunk)
        if chunks.endswith(b"\n"):
            break
        if len(chunks) > max_bytes:
            raise _field_error("JSON line exceeds maximum size")
    line = bytes(chunks).strip()
    if not line:
        raise _field_error("empty JSON line")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _field_error(f"invalid JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise _field_error("JSON response must be an object")
    return value


def validate_fields(message: dict[str, Any], fields: Iterable[str], *, port: int | None = None) -> None:
    missing = [field for field in fields if field not in message]
    if missing:
        raise _field_error(
            f"response is missing required fields: {missing}",
            port=port,
            details={"missing": missing, "received_keys": sorted(message)},
        )


def rpc(
    host: str,
    port: int,
    message: dict[str, Any],
    timeout_ms: int = 3000,
    *,
    required_fields: Iterable[str] = (),
) -> dict[str, Any]:
    timeout = max(0.05, timeout_ms / 1000.0)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(encode_message(message))
            response = read_json_line(sock)
    except (TimeoutError, socket.timeout) as exc:
        raise _timeout_error("reading response from", host, port) from exc
    except ConnectionRefusedError as exc:
        raise ShardMeshError(
            "E_PORT",
            f"connection refused at {host}:{port}",
            category="node",
            field_name="port",
            details={"host": host, "port": port},
        ) from exc
    except OSError as exc:
        if exc.errno in {errno.ETIMEDOUT, errno.EWOULDBLOCK, errno.EINPROGRESS}:
            raise _timeout_error("socket operation at", host, port) from exc
        raise ShardMeshError(
            "E_NODE",
            f"socket failure at {host}:{port}: {exc}",
            category="node",
            field_name="socket",
            details={"host": host, "port": port, "errno": exc.errno},
        ) from exc

    if response.get("ok") is False:
        code = str(response.get("code", "E_NODE"))
        if code in {"E_SPEC", "E_PORT", "E_TIMEOUT", "E_FIELD", "E_LEASE", "E_RENEW", "E_DIVERGE", "E_REPLAY"}:
            raise ShardMeshError(
                code,
                str(response.get("message", "node rejected request")),
                category=response.get("category"),
                field_name=response.get("field"),
                details=response.get("details", {}),
            )
        raise ShardMeshError(code, str(response.get("message", "node error")), details=response.get("details", {}))
    validate_fields(response, required_fields, port=port)
    return response


def handshake(host: str, port: int, node_id: str, timeout_ms: int = 1000) -> dict[str, Any]:
    response = rpc(
        host,
        port,
        {"type": "hello", "node_id": node_id, "protocol_version": 1},
        timeout_ms,
        required_fields=("type", "node_id", "protocol_version"),
    )
    if response.get("type") != "hello_ack":
        raise _field_error("handshake response type must be hello_ack", port=port)
    if response.get("node_id") != node_id:
        raise _field_error(
            f"handshake node id mismatch: expected {node_id!r}, got {response.get('node_id')!r}",
            port=port,
        )
    return response

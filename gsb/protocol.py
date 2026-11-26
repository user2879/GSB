from __future__ import annotations

import json
import socket
from typing import Any

from .errors import (
    GSBError,
    LeaseError,
    PortError,
    ProtocolError,
    ProtocolTimeout,
    VerifyError,
    DivergeError,
)


MAX_LINE_BYTES = 64 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def recv_line(sock: socket.socket, *, max_bytes: int = MAX_LINE_BYTES) -> dict[str, Any]:
    buffer = bytearray()
    while True:
        try:
            chunk = sock.recv(65536)
        except (socket.timeout, TimeoutError) as exc:
            raise ProtocolTimeout("timed out while reading JSON line", field="socket") from exc
        except OSError as exc:
            raise ProtocolError(f"socket read failed: {exc}", field="socket") from exc
        if not chunk:
            if not buffer:
                raise ProtocolError("peer closed connection without a JSON line", field="socket")
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise ProtocolError("JSON line exceeds maximum size", field="socket")
        newline = buffer.find(b"\n")
        if newline >= 0:
            line = bytes(buffer[:newline])
            return _decode_line(line)
    return _decode_line(bytes(buffer))


def _decode_line(line: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON line: {exc}", field="socket") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("JSON line must be an object", field="socket")
    if payload.get("ok") is False:
        raise remote_error(payload)
    return payload


def remote_error(payload: dict[str, Any]) -> GSBError:
    code = payload.get("code", "E_PROTOCOL")
    message = str(payload.get("error", "remote node error"))
    field = payload.get("field")
    position = payload.get("position")
    error_type = {
        "E_LEASE": LeaseError,
        "E_PORT": PortError,
        "E_TIMEOUT": ProtocolTimeout,
        "E_DIVERGE": DivergeError,
        "E_VERIFY": VerifyError,
    }.get(code, ProtocolError)
    return error_type(message, field=field, position=position, remote=payload)


def call_jsonline(
    host: str,
    port: int,
    message: dict[str, Any],
    *,
    timeout_ms: int = 3000,
) -> dict[str, Any]:
    timeout = max(0.001, timeout_ms / 1000.0)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            send_message(sock, message)
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            return recv_line(sock)
    except ProtocolTimeout:
        raise
    except (socket.timeout, TimeoutError) as exc:
        raise ProtocolTimeout(f"timed out connecting to {host}:{port}", field="socket") from exc
    except ConnectionRefusedError as exc:
        raise ProtocolError(f"connection refused at {host}:{port}", field="port", position=port) from exc
    except OSError as exc:
        raise ProtocolError(f"TCP call failed at {host}:{port}: {exc}", field="socket") from exc


def check_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise PortError(
                f"loopback port {port} is occupied or cannot bind: {exc}",
                field="nodes.port",
                position=port,
            ) from exc


def wait_for_tcp(host: str, port: int, *, timeout_ms: int = 5000) -> None:
    deadline_ns = time_monotonic_ns() + timeout_ms * 1_000_000
    last_error: Exception | None = None
    while time_monotonic_ns() < deadline_ns:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError as exc:
            last_error = exc
    raise ProtocolTimeout(
        f"node at {host}:{port} did not open within {timeout_ms}ms",
        field="nodes.port",
        position=port,
        last_error=str(last_error),
    )


def time_monotonic_ns() -> int:
    import time

    return time.monotonic_ns()

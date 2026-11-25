"""Line-delimited JSON over loopback TCP, with real socket timeouts."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from .canonical import canonical_bytes
from .errors import DiagnosticError


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 64 * 1024 * 1024


def send_json_line(sock: socket.socket, value: Any) -> None:
    sock.sendall(canonical_bytes(value) + b"\n")


class LineReader:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buffer = bytearray()

    def receive(self, timeout_ms: int, *, phase: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                if len(line) > MAX_LINE_BYTES:
                    raise DiagnosticError(
                        "E_FIELD",
                        "protocol line exceeds maximum size",
                        field="line",
                        details={"phase": phase, "bytes": len(line)},
                    )
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise DiagnosticError(
                        "E_FIELD",
                        f"invalid protocol JSON: {exc}",
                        field="line",
                        details={"phase": phase},
                    ) from exc
                if not isinstance(value, dict):
                    raise DiagnosticError(
                        "E_FIELD",
                        "protocol message must be a JSON object",
                        field="$",
                        details={"phase": phase},
                    )
                if value.get("type") == "error":
                    error = value.get("error") if isinstance(value.get("error"), dict) else {}
                    raise DiagnosticError(
                        str(error.get("code", "E_FIELD")),
                        str(error.get("message", "remote worker error")),
                        field=error.get("field"),
                        line=error.get("line"),
                        details={"phase": phase, "remote": True, **dict(error.get("details") or {})},
                    )
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiagnosticError(
                    "E_HANDSHAKE",
                    "timed out waiting for a loopback protocol response",
                    field="socket.recv",
                    details={"phase": phase, "timeout_ms": timeout_ms, "timeout_source": "socket.recv"},
                )
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout as exc:
                raise DiagnosticError(
                    "E_HANDSHAKE",
                    "timed out waiting for a loopback protocol response",
                    field="socket.recv",
                    details={"phase": phase, "timeout_ms": timeout_ms, "timeout_source": "socket.recv"},
                ) from exc
            except OSError as exc:
                raise DiagnosticError(
                    "E_HANDSHAKE",
                    f"loopback socket failed while waiting: {exc}",
                    field="socket.recv",
                    details={"phase": phase, "timeout_source": "socket.recv"},
                ) from exc
            if not chunk:
                raise DiagnosticError(
                    "E_HANDSHAKE",
                    "worker closed the loopback connection before replying",
                    field="socket.recv",
                    details={"phase": phase, "timeout_source": "socket.recv"},
                )
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_LINE_BYTES:
                raise DiagnosticError(
                    "E_FIELD",
                    "protocol buffer exceeds maximum line size",
                    field="line",
                    details={"phase": phase},
                )


def require_fields(value: dict[str, Any], fields: tuple[str, ...], *, phase: str) -> None:
    for field in fields:
        if field not in value:
            raise DiagnosticError(
                "E_FIELD",
                "required protocol field is missing",
                field=field,
                details={"phase": phase},
            )


def perform_handshake(
    sock: socket.socket,
    reader: LineReader,
    *,
    expected_worker_id: str,
    expected_param_version: int,
    expected_param_digest: str,
    timeout_ms: int,
    phase: str = "handshake",
) -> dict[str, Any]:
    send_json_line(
        sock,
        {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "worker_id": expected_worker_id,
        },
    )
    ready = reader.receive(timeout_ms, phase=phase)
    try:
        require_fields(
            ready,
            ("type", "worker_id", "param_version", "param_digest"),
            phase=phase,
        )
    except DiagnosticError as exc:
        exc.details.update(
            {
                "reported_keys": sorted(ready),
                "reported_message": ready,
                "expected_worker_id": expected_worker_id,
                "peer": str(sock.getpeername()) if sock.fileno() >= 0 else None,
            }
        )
        raise
    if ready["type"] != "ready":
        raise DiagnosticError(
            "E_FIELD",
            "expected a ready message",
            field="type",
            details={"phase": phase, "value": ready["type"]},
        )
    if ready["worker_id"] != expected_worker_id:
        raise DiagnosticError(
            "E_FIELD",
            "worker id does not match the requested worker",
            field="worker_id",
            details={"phase": phase, "expected": expected_worker_id, "actual": ready["worker_id"]},
        )
    if ready["param_version"] != expected_param_version:
        raise DiagnosticError(
            "E_FIELD",
            "worker parameter version does not match the request",
            field="param_version",
            details={"phase": phase, "expected": expected_param_version, "actual": ready["param_version"]},
        )
    if ready["param_digest"] != expected_param_digest:
        raise DiagnosticError(
            "E_FIELD",
            "worker parameter digest does not match the request",
            field="param_digest",
            details={"phase": phase},
        )
    return ready

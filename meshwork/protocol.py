from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import MeshError
from .model import LOOPBACK_HOST, RequestSpec

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 1_048_576


@dataclass
class AttemptResult:
    outcome: str
    transport_ok: bool
    status: str | None = None
    error_code: str | None = None
    latency_ms: int = 0
    upstream_delay_ms: int = 0
    detail: str | None = None
    reply: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.transport_ok and self.status == "ok" and self.error_code is None


def _mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _recv_line(sock: socket.socket) -> bytes:
    chunks = bytearray()
    while len(chunks) <= MAX_LINE_BYTES:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.extend(chunk)
        if b"\n" in chunk:
            break
    if len(chunks) > MAX_LINE_BYTES:
        raise MeshError("E_REPLY", "reply line exceeds maximum length", field="reply.line")
    line, _, _ = bytes(chunks).partition(b"\n")
    return line


def _decode_reply(line: bytes) -> dict[str, Any]:
    if not line:
        raise MeshError("E_REPLY", "empty reply line", field="reply")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshError("E_REPLY", f"invalid JSON reply: {exc}", field="reply") from exc
    if not isinstance(value, dict):
        raise MeshError("E_REPLY", "reply must be a JSON object", field="reply")
    return value


def _require_reply_fields(reply: dict[str, Any], fields: list[str]) -> None:
    for name in fields:
        if name not in reply:
            raise MeshError(
                "E_REPLY",
                f"reply is missing required field {name!r}",
                field=f"reply.{name}",
            )


def send_ping(
    host: str,
    port: int,
    instance_id: str,
    timeout_ms: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {"protocol": PROTOCOL_VERSION, "type": "ping"},
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    try:
        with socket.create_connection((host, port), timeout=timeout_ms / 1000.0) as sock:
            sock.settimeout(timeout_ms / 1000.0)
            sock.sendall(payload)
            reply = _decode_reply(_recv_line(sock))
    except (TimeoutError, socket.timeout) as exc:
        raise MeshError(
            "E_HOST",
            f"handshake timed out after {timeout_ms} ms",
            field="handshake.timeout_ms",
            details={"host": host, "port": port, "instance_id": instance_id},
        ) from exc
    except OSError as exc:
        raise MeshError(
            "E_HOST",
            f"handshake connection failed: {exc}",
            field="handshake.connect",
            details={"host": host, "port": port, "instance_id": instance_id},
        ) from exc
    _require_reply_fields(
        reply,
        ["protocol", "type", "instance_id", "service_id", "pid"],
    )
    if reply["type"] != "pong":
        raise MeshError("E_REPLY", "handshake reply type must be 'pong'", field="reply.type")
    if reply["protocol"] != PROTOCOL_VERSION:
        raise MeshError(
            "E_REPLY",
            f"unsupported upstream protocol {reply['protocol']!r}",
            field="reply.protocol",
        )
    if reply["instance_id"] != instance_id:
        raise MeshError(
            "E_REPLY",
            f"handshake instance_id mismatch: expected {instance_id!r}, got {reply['instance_id']!r}",
            field="reply.instance_id",
        )
    return reply


def send_attempt(
    instance_host: str,
    instance_port: int,
    instance_id: str,
    service_id: str,
    request: RequestSpec,
    attempt_no: int,
    selection_seq: int,
    route_revision: int,
    timeout_ms: int,
) -> AttemptResult:
    request_payload = {
        "protocol": PROTOCOL_VERSION,
        "type": "request",
        "request_id": request.id,
        "service_id": service_id,
        "instance_id": instance_id,
        "attempt_no": attempt_no,
        "selection_seq": selection_seq,
        "route_revision": route_revision,
        "payload": request.payload,
    }
    wire = json.dumps(
        request_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    started = _mono_ms()
    try:
        with socket.create_connection(
            (instance_host, instance_port), timeout=timeout_ms / 1000.0
        ) as sock:
            sock.settimeout(timeout_ms / 1000.0)
            sock.sendall(wire)
            line = _recv_line(sock)
    except (TimeoutError, socket.timeout) as exc:
        elapsed = max(0, _mono_ms() - started)
        return AttemptResult(
            outcome="timeout",
            transport_ok=False,
            error_code="E_TIMEOUT",
            latency_ms=elapsed,
            detail=f"real socket timeout after {timeout_ms} ms",
        )
    except OSError as exc:
        elapsed = max(0, _mono_ms() - started)
        return AttemptResult(
            outcome="host_error",
            transport_ok=False,
            error_code="E_HOST",
            latency_ms=elapsed,
            detail=str(exc),
        )
    elapsed = max(0, _mono_ms() - started)
    try:
        reply = _decode_reply(line)
        _require_reply_fields(
            reply,
            [
                "protocol",
                "request_id",
                "attempt_no",
                "instance_id",
                "status",
                "latency_class",
                "delay_ms",
            ],
        )
    except MeshError as exc:
        return AttemptResult(
            outcome="reply_error",
            transport_ok=True,
            error_code="E_REPLY",
            latency_ms=elapsed,
            detail=exc.message,
            reply={"parse_error": exc.to_dict()},
        )
    mismatches: list[str] = []
    if reply.get("request_id") != request.id:
        mismatches.append("request_id")
    if reply.get("attempt_no") != attempt_no:
        mismatches.append("attempt_no")
    if reply.get("instance_id") != instance_id:
        mismatches.append("instance_id")
    if reply.get("protocol") != PROTOCOL_VERSION:
        mismatches.append("protocol")
    if reply.get("status") not in {"ok", "error"}:
        mismatches.append("status")
    if reply.get("latency_class") not in {"fast", "slow"}:
        mismatches.append("latency_class")
    if isinstance(reply.get("delay_ms"), bool) or not isinstance(reply.get("delay_ms"), int):
        mismatches.append("delay_ms")
    if mismatches:
        return AttemptResult(
            outcome="reply_error",
            transport_ok=True,
            error_code="E_REPLY",
            latency_ms=elapsed,
            detail="reply field mismatch: " + ", ".join(mismatches),
            reply=reply,
        )
    status = str(reply["status"])
    outcome = "success" if status == "ok" else "error"
    return AttemptResult(
        outcome=outcome,
        transport_ok=True,
        status=status,
        error_code=None if status == "ok" else "E_UPSTREAM",
        latency_ms=elapsed,
        upstream_delay_ms=int(reply["delay_ms"]),
        detail=str(reply.get("detail", "")),
        reply=reply,
    )


def check_port_free(host: str, port: int) -> None:
    if host != LOOPBACK_HOST:
        raise MeshError("E_SPEC", "only 127.0.0.1 is allowed", field="host", location=host)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise MeshError(
            "E_PORT",
            f"port {port} on {host} is not available: {exc}",
            field="port",
            location=f"{host}:{port}",
            details={"host": host, "port": port},
        ) from exc
    finally:
        probe.close()

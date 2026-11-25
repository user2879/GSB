from __future__ import annotations

import json
import socket
from typing import Any, Optional

from .errors import E_FIELD, E_PORT, E_TIMEOUT, MeshError
from .timeutil import monotonic_ms


MAX_LINE_BYTES = 4 * 1024 * 1024


def label_for(lease_id: str, generation: int) -> str:
    return f"{lease_id}@{int(generation)}"


def generation_from_label(label: Any, lease_id: Optional[str] = None) -> int:
    if not isinstance(label, str) or "@" not in label:
        raise MeshError("label must have form <lease_id>@<generation>", code=E_FIELD, field="label")
    prefix, suffix = label.rsplit("@", 1)
    if lease_id is not None and prefix != lease_id:
        raise MeshError("label lease_id mismatch", code=E_FIELD, field="label")
    if not suffix.isdigit() or int(suffix) <= 0:
        raise MeshError("label generation must be a positive integer", code=E_FIELD, field="label")
    return int(suffix)


def encode_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _recv_until_newline(sock: socket.socket, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise MeshError("connection closed before a complete JSON line", code=E_PORT)
        chunks.append(chunk)
        size += len(chunk)
        if b"\n" in chunk:
            break
        if size > max_bytes:
            raise MeshError("JSON line exceeds buffer limit", code=E_FIELD, field="line")
    return b"".join(chunks).split(b"\n", 1)[0]


def read_json_line(sock: socket.socket, max_bytes: int = MAX_LINE_BYTES) -> dict[str, Any]:
    raw = _recv_until_newline(sock, max_bytes)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshError(f"invalid JSON-line response: {exc}", code=E_FIELD, field="response") from exc
    if not isinstance(payload, dict):
        raise MeshError("JSON-line response must be an object", code=E_FIELD, field="response")
    return payload


def _set_deadline_timeout(sock: socket.socket, deadline_ms: Optional[float], default_ms: int) -> None:
    if deadline_ms is None:
        sock.settimeout(max(0.001, default_ms / 1000.0))
        return
    remaining_s = max(0.001, (deadline_ms - monotonic_ms()) / 1000.0)
    sock.settimeout(remaining_s)


def request_json_line(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    node_id: Optional[str] = None,
    timeout_ms: int = 2000,
    handshake_timeout_ms: Optional[int] = None,
    buffer_bytes: int = MAX_LINE_BYTES,
    validate_receipt: bool = True,
) -> dict[str, Any]:
    """Connect, perform a real hello handshake, and exchange one JSON line."""
    handshake_ms = handshake_timeout_ms if handshake_timeout_ms is not None else timeout_ms
    deadline = monotonic_ms() + max(1, timeout_ms)
    sock: Optional[socket.socket] = None
    try:
        sock = socket.create_connection((host, port), timeout=max(0.001, handshake_ms / 1000.0))
    except socket.timeout as exc:
        raise MeshError(
            f"TCP connect/handshake timed out for {host}:{port}",
            code=E_TIMEOUT,
            field="handshake",
            details={"node_id": node_id, "port": port},
        ) from exc
    except ConnectionRefusedError as exc:
        raise MeshError(
            f"connection refused on loopback port {port}",
            code=E_PORT,
            field="port",
            details={"node_id": node_id, "port": port},
        ) from exc
    except OSError as exc:
        code = E_TIMEOUT if isinstance(exc, socket.timeout) else E_PORT
        raise MeshError(str(exc), code=code, field="port") from exc

    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _set_deadline_timeout(sock, None, handshake_ms)
        hello = {"op": "hello", "client": "leasemesh-center/1", "node_id": node_id}
        sock.sendall(encode_line(hello))
        try:
            handshake = read_json_line(sock, buffer_bytes)
        except socket.timeout as exc:
            raise MeshError(
                "handshake read timed out",
                code=E_TIMEOUT,
                field="handshake",
                details={"node_id": node_id, "port": port},
            ) from exc
        if not handshake.get("ok") or handshake.get("type") != "hello_ack":
            raise MeshError("invalid handshake acknowledgement", code=E_FIELD, field="handshake")
        if node_id is not None and handshake.get("node_id") != node_id:
            raise MeshError("handshake node_id mismatch", code=E_FIELD, field="node_id")

        _set_deadline_timeout(sock, deadline, timeout_ms)
        sent_at = monotonic_ms()
        sock.sendall(encode_line(payload))
        try:
            response = read_json_line(sock, buffer_bytes)
        except socket.timeout as exc:
            raise MeshError(
                "response read timed out",
                code=E_TIMEOUT,
                field="response",
                details={"node_id": node_id, "port": port, "waited_ms": monotonic_ms() - sent_at},
            ) from exc

        if response.get("ok") is False:
            code = response.get("code") or "E_LEASE"
            if code not in {"E_LEASE", "E_FIELD", "E_TIMEOUT", "E_PORT", "E_SPEC", "E_STATE"}:
                code = "E_LEASE"
            raise MeshError(
                str(response.get("error") or "node rejected request"),
                code=code,
                field=response.get("field"),
                details=response,
            )

        if validate_receipt and payload.get("op") in {"grant", "renew", "release"}:
            required = ("node_id", "lease_id", "generation", "checksum_digest", "compute_ms")
            missing = [key for key in required if key not in response]
            if missing:
                raise MeshError(
                    "receipt is missing required fields",
                    code=E_FIELD,
                    field=missing[0],
                    details={"missing": missing, "response": response},
                )
            if node_id is not None and response.get("node_id") != node_id:
                raise MeshError("receipt node_id does not match target", code=E_FIELD, field="node_id")
            if response.get("lease_id") != payload.get("lease_id"):
                raise MeshError("receipt lease_id mismatch", code=E_FIELD, field="lease_id")
        return response
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

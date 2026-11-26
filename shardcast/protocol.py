"""JSON-lines TCP protocol primitives with real socket timeouts."""

from __future__ import annotations

import json
import socket

from .errors import FieldError, PortError, TimeoutError

HANDSHAKE_HELLO = {"type": "hello", "protocol": "shardcast-jsonl/1"}
HANDSHAKE_READY = {"type": "ready", "protocol": "shardcast-jsonl/1"}

DISPATCH_REQUIRED = (
    "type",
    "lease_id",
    "lease_version",
    "node_id",
    "shard_id",
    "capacity_version",
    "sample_rows",
    "batches",
)

RECEIPT_REQUIRED = (
    "type",
    "node_id",
    "shard_id",
    "capacity_version",
    "lease_id",
    "lease_version",
    "summary",
    "elapsed_ms",
)


def write_line(sock: socket.socket, payload: dict) -> None:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sock.sendall(raw + b"\n")


def read_line(sock: socket.socket, max_bytes: int = 64 * 1024 * 1024) -> dict:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout as exc:
            raise TimeoutError("timed out while reading a JSON line", path="socket") from exc
        if not chunk:
            raise TimeoutError("peer closed connection before a full line arrived", path="socket")
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            break
        if total > max_bytes:
            raise FieldError("incoming JSON line exceeds size limit", path="socket")
    raw = b"".join(chunks)
    line, _, _ = raw.partition(b"\n")
    try:
        payload = json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FieldError("response is not valid JSON", path="socket") from exc
    if not isinstance(payload, dict):
        raise FieldError("response payload must be a JSON object", path="socket")
    return payload


def require_fields(payload: dict, required: tuple[str, ...], context: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise FieldError(
            f"missing receipt fields: {', '.join(missing)}",
            path=f"{context}.{missing[0]}",
            detail={"missing": missing},
        )


def connect_and_handshake(
    host: str, port: int, timeout: float
) -> socket.socket:
    """Open a real TCP socket and perform the two-message handshake."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError as exc:
        sock.close()
        raise PortError(f"connection refused on {host}:{port}", path=f"{host}:{port}") from exc
    except OSError as exc:
        sock.close()
        raise PortError(f"cannot connect to {host}:{port}: {exc}", path=f"{host}:{port}") from exc
    try:
        write_line(sock, HANDSHAKE_HELLO)
        ready = read_line(sock)
    except TimeoutError:
        sock.close()
        raise
    if ready.get("type") != "ready":
        sock.close()
        raise FieldError("handshake did not return a ready message", path="handshake.type")
    return sock


def request_dispatch(
    host: str,
    port: int,
    request: dict,
    timeout: float,
) -> dict:
    """One socket per dispatch: handshake, send one request, read one receipt."""
    require_fields(request, DISPATCH_REQUIRED, "request")
    try:
        sock = connect_and_handshake(host, port, timeout)
    except TimeoutError as exc:
        raise TimeoutError(
            f"handshake timeout for shard {request.get('shard_id')}",
            path=f"{host}:{port}",
        ) from exc
    try:
        write_line(sock, request)
        receipt = read_line(sock)
    finally:
        sock.close()
    require_fields(receipt, RECEIPT_REQUIRED, "receipt")
    for identity_field in ("node_id", "shard_id", "lease_id"):
        if receipt[identity_field] != request[identity_field]:
            raise FieldError(
                f"receipt {identity_field} does not match request",
                path=f"receipt.{identity_field}",
            )
    if receipt["capacity_version"] != request["capacity_version"]:
        raise FieldError("receipt capacity_version does not match request", path="receipt.capacity_version")
    return receipt


def request_renew(host: str, port: int, request: dict, timeout: float) -> dict:
    required = ("type", "lease_id", "lease_version", "node_id", "shard_id", "capacity_version")
    require_fields(request, required, "renew_request")
    try:
        sock = connect_and_handshake(host, port, timeout)
    except TimeoutError as exc:
        raise TimeoutError(
            f"renew handshake timeout for lease {request.get('lease_id')}",
            path=f"{host}:{port}",
        ) from exc
    try:
        write_line(sock, request)
        receipt = read_line(sock)
    finally:
        sock.close()
    require_fields(
        receipt,
        ("type", "lease_id", "lease_version", "node_id", "shard_id", "accepted", "expire_ms"),
        "renew_receipt",
    )
    return receipt

from __future__ import annotations

import json
import socket
import time
from typing import Any, Mapping

from .errors import FieldError, TimeoutError


PROTOCOL_NAME = "hopledger-jsonl/1"
REQUIRED_RECEIPT_FIELDS = (
    "schema",
    "type",
    "hop_id",
    "node_id",
    "request_id",
    "config_version",
    "scale_factor",
    "monotonic_ms",
)


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def monotonic_now_ns() -> int:
    return time.monotonic_ns()


def encode_line(message: Mapping[str, Any]) -> bytes:
    return json.dumps(message, sort_keys=True, separators=(",", ":")) .encode("utf-8") + b"\n"


def read_jsonline(sock: socket.socket, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    chunks = bytearray()
    while True:
        char = sock.recv(1)
        if not char:
            raise FieldError("socket closed before a complete JSON line", field="line")
        if char == b"\n":
            break
        chunks.extend(char)
        if len(chunks) > max_bytes:
            raise FieldError("JSON line exceeds maximum size", field="line")
    try:
        value = json.loads(chunks.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FieldError(f"invalid JSON line: {exc}", field="line") from exc
    if not isinstance(value, dict):
        raise FieldError("JSON line must be an object", field="line")
    return value


def _connect(host: str, port: int, timeout_seconds: float) -> socket.socket:
    try:
        return socket.create_connection((host, port), timeout=timeout_seconds)
    except socket.timeout as exc:
        raise TimeoutError(
            f"TCP connect/handshake timed out after {timeout_seconds:.3f}s",
            host=host,
            port=port,
            field="handshake",
        ) from exc
    except TimeoutError as exc:  # pragma: no cover - platform socket alias
        raise TimeoutError(
            f"TCP connect/handshake timed out after {timeout_seconds:.3f}s",
            host=host,
            port=port,
            field="handshake",
        ) from exc


def read_handshake_ack(sock: socket.socket, timeout_seconds: float) -> dict[str, Any]:
    sock.settimeout(timeout_seconds)
    try:
        message = read_jsonline(sock)
    except socket.timeout as exc:
        raise TimeoutError(
            f"handshake read timed out after {timeout_seconds:.3f}s",
            field="handshake",
        ) from exc
    if message.get("type") != "hello_ack" or message.get("protocol") != PROTOCOL_NAME:
        raise FieldError("invalid handshake acknowledgement", field="hello_ack",
                         received=message)
    return message


def handshake(
    port: int,
    *,
    host: str = "127.0.0.1",
    timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    """Perform a real TCP connect plus blocking handshake read."""

    sock = _connect(host, port, timeout_seconds)
    try:
        return read_handshake_ack(sock, timeout_seconds)
    finally:
        sock.close()


def validate_receipt(receipt: Any, *, expected_node_id: str | None = None,
                     expected_request_id: str | None = None) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise FieldError("receipt must be a JSON object", hop_id=expected_node_id,
                         field="receipt")
    for field_name in REQUIRED_RECEIPT_FIELDS:
        if field_name not in receipt:
            raise FieldError(
                f"receipt is missing required field {field_name!r}",
                hop_id=receipt.get("node_id", expected_node_id),
                request_id=receipt.get("request_id", expected_request_id),
                field=field_name,
            )
    if expected_node_id is not None and receipt["node_id"] != expected_node_id:
        raise FieldError(
            "receipt node_id does not match addressed node",
            hop_id=receipt.get("node_id"),
            field="node_id",
            expected=expected_node_id,
        )
    if receipt.get("hop_id") != receipt.get("node_id"):
        raise FieldError(
            "receipt hop_id and node_id must identify the same chain node",
            hop_id=receipt.get("hop_id"),
            field="hop_id",
            node_id=receipt.get("node_id"),
        )
    if expected_request_id is not None and receipt["request_id"] != expected_request_id:
        raise FieldError(
            "receipt request_id does not match request",
            hop_id=receipt.get("node_id"),
            request_id=receipt.get("request_id"),
            field="request_id",
            expected=expected_request_id,
        )
    return receipt


def call_jsonl(
    *,
    host: str = "127.0.0.1",
    port: int,
    message: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Open a real loopback socket, handshake, send one call, read one reply."""

    sock = _connect(host, port, timeout_seconds)
    try:
        read_handshake_ack(sock, timeout_seconds)
        sock.settimeout(timeout_seconds)
        try:
            sock.sendall(encode_line(message))
            response = read_jsonline(sock)
        except socket.timeout as exc:
            raise TimeoutError(
                f"JSONL response timed out after {timeout_seconds:.3f}s",
                port=port,
                field="response",
            ) from exc
        validate_receipt(
            response.get("receipt"),
            expected_node_id=message.get("node_id"),
            expected_request_id=message.get("request_id"),
        )
        return response
    finally:
        try:
            sock.close()
        except OSError:
            pass

"""Line-based NDJSON wire protocol shared by parent and service subprocess."""
from __future__ import annotations

import json
import socket
from typing import Any

from .errors import ReplyError, TimeoutError as DomainTimeoutError

REQUIRED_REPLY_FIELDS = ("type", "env_id", "seq_low", "seq_high",
                         "authority_digest", "elapsed_ms")
BATCH_REQUIRED_FIELDS = ("type", "env_id", "accepted", "stored", "keys")


def send_line(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n").encode())


def recv_line(sock: socket.socket, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            byte = sock.recv(1)
        except socket.timeout as exc:
            raise DomainTimeoutError(
                "socket timeout while reading reply", field="<wire>"
            ) from exc
        if not byte:
            raise ConnectionError("peer closed connection before newline")
        if byte == b"\n":
            break
        chunks.append(byte)
        total += 1
        if total > max_bytes:
            raise ReplyError("reply line exceeds maximum size", field="<wire>")
    return json.loads(b"".join(chunks).decode("utf-8"))


def validate_reply(reply: Any) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ReplyError("reply must be a JSON object", field="<reply>")
    required = (BATCH_REQUIRED_FIELDS if reply.get("type") == "batch_ack"
                else REQUIRED_REPLY_FIELDS)
    missing = [name for name in required if name not in reply]
    if missing:
        raise ReplyError(
            f"reply missing required fields: {','.join(missing)}",
            field=f"<reply>.{missing[0]}",
        )
    return reply

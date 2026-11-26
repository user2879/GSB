"""Newline-delimited JSON over real TCP sockets with deadline enforcement."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from .errors import GSBTimeoutError, ProtocolError

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024 * 1024


def monotonic_ms_since(start_ns: int) -> int:
    return max(0, (time.monotonic_ns() - start_ns) // 1_000_000)


class JsonLineChannel:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._buffer = bytearray()

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def send(self, value: dict[str, Any]) -> None:
        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as exc:
            raise ProtocolError("response is not JSON serializable", details={"error": str(exc)}) from exc
        if len(payload) > MAX_LINE_BYTES:
            raise ProtocolError(
                "outgoing JSON line exceeds the protocol limit",
                details={"bytes": len(payload), "limit": MAX_LINE_BYTES},
            )
        try:
            self.sock.sendall(payload)
        except socket.timeout as exc:
            raise GSBTimeoutError("socket write timed out", details={"stage": "write"}) from exc
        except OSError as exc:
            raise ProtocolError("socket write failed", details={"error": str(exc)}) from exc

    def receive(self, timeout_s: float, *, stage: str = "read") -> dict[str, Any]:
        if timeout_s <= 0:
            raise GSBTimeoutError("socket deadline is not positive", details={"stage": stage})
        deadline = time.monotonic() + timeout_s
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                if not line:
                    raise ProtocolError("empty JSON line", details={"stage": stage})
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProtocolError(
                        "invalid JSON line",
                        details={"stage": stage, "error": str(exc)},
                    ) from exc
                if not isinstance(value, dict):
                    raise ProtocolError("protocol message must be a JSON object", details={"stage": stage})
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GSBTimeoutError(
                    "socket read timed out",
                    details={"stage": stage, "timeout_s": timeout_s},
                )
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(65_536)
            except socket.timeout as exc:
                raise GSBTimeoutError(
                    "socket read timed out",
                    details={"stage": stage, "timeout_s": timeout_s},
                ) from exc
            except OSError as exc:
                raise ProtocolError("socket read failed", details={"stage": stage, "error": str(exc)}) from exc
            if not chunk:
                raise ProtocolError("peer closed the socket before a complete JSON line", details={"stage": stage})
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_LINE_BYTES:
                raise ProtocolError(
                    "incoming JSON line exceeds the protocol limit",
                    details={"limit": MAX_LINE_BYTES, "stage": stage},
                )


def require_fields(message: dict[str, Any], fields: tuple[str, ...], *, stage: str) -> None:
    missing = [field for field in fields if field not in message or message[field] is None]
    if missing:
        raise ProtocolError(
            "protocol message is missing required fields",
            details={"stage": stage, "missing": missing},
        )


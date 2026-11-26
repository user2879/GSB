"""JSON-lines TCP client and framing helpers."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from .errors import E_PROTOCOL, QLabError

MAX_LINE_BYTES = 32 * 1024 * 1024


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def decode_message(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QLabError(E_PROTOCOL, f"invalid JSON message: {exc}") from exc
    if not isinstance(value, dict):
        raise QLabError(E_PROTOCOL, "protocol message must be an object")
    return value


class JsonLineClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, timeout: float = 5.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def request(self, message: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        payload = encode_message(message)
        with socket.create_connection((self.host, self.port), timeout=timeout or self.timeout) as sock:
            sock.settimeout(timeout or self.timeout)
            sock.sendall(payload)
            chunks = bytearray()
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.extend(chunk)
                newline = chunks.find(b"\n")
                if newline >= 0:
                    return decode_message(bytes(chunks[:newline]))
                if len(chunks) > MAX_LINE_BYTES:
                    raise QLabError(E_PROTOCOL, "response line exceeds size limit")
        if not chunks:
            raise QLabError(E_PROTOCOL, "empty protocol response")
        return decode_message(bytes(chunks))

    def wait_ready(self, *, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = self.request({"op": "ping", "request_id": "readiness"}, timeout=0.5)
                if response.get("ok"):
                    return response
            except (OSError, QLabError) as exc:
                last_error = exc
            time.sleep(0.02)
        raise QLabError(E_PROTOCOL, f"worker at {self.host}:{self.port} did not become ready: {last_error}")


def response_ok(response: dict[str, Any]) -> bool:
    return bool(response.get("ok", False))


def response_code(response: dict[str, Any]) -> str | None:
    code = response.get("code")
    return str(code) if code is not None else None

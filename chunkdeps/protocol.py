"""Line-delimited JSON protocol shared by the service and local clients."""

from __future__ import annotations

import json
import socket
import uuid
from typing import Any

from .errors import E_CONFLICT, E_INPUT, BusinessError
from .util import canonical_text


PROTOCOL_NAME = "chunkdeps-jsonl/v1"
MAX_FRAME_BYTES = 32 * 1024 * 1024


class RemoteBusinessError(BusinessError):
    pass


class ProtocolClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 30.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, op: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = "req-" + uuid.uuid4().hex[:20]
        frame = {
            "protocol": PROTOCOL_NAME,
            "request_id": request_id,
            "op": op,
            "params": params or {},
        }
        wire = (canonical_text(frame) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(wire)
                response_file = sock.makefile("rb")
                line = response_file.readline(MAX_FRAME_BYTES + 1)
        except OSError as exc:
            raise BusinessError(
                E_CONFLICT,
                "cannot reach the local storage service",
                {"host": self.host, "port": self.port, "reason": str(exc)},
            ) from exc
        if not line:
            raise BusinessError(
                E_CONFLICT,
                "storage service closed the connection without a response",
                {"host": self.host, "port": self.port},
            )
        if len(line) > MAX_FRAME_BYTES:
            raise BusinessError(E_INPUT, "response frame is too large")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BusinessError(E_CONFLICT, "storage service returned invalid JSON") from exc
        if response.get("request_id") != request_id:
            raise BusinessError(
                E_CONFLICT,
                "response request id mismatch",
                {"expected": request_id, "actual": response.get("request_id")},
            )
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RemoteBusinessError(
                str(error.get("code", "E_INTERNAL")),
                str(error.get("message", "remote request failed")),
                error.get("details") or {},
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise BusinessError(E_CONFLICT, "storage service returned a non-object result")
        return result

    def ping(self) -> dict[str, Any]:
        return self.request("ping", {})


from __future__ import annotations

import json
import socket
from typing import Any

from .errors import FieldError, ProtocolTimeout, StaleGenerationError


PROTOCOL_VERSION = 1

REQUIRED_RESPONSE_FIELDS: dict[str, tuple[str, ...]] = {
    "grant": ("lease_id", "generation", "state"),
    "renew": ("lease_id", "generation", "state"),
    "release": ("lease_id", "generation", "state"),
    "expire": ("lease_id", "generation", "state", "reaped"),
    "write": ("lease_id", "shard_index", "digest"),
    "read": ("lease_id", "shard_index", "digest", "content_b64"),
    "status": ("node_id", "leases"),
}


def encode_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


class RemoteProtocolError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class JsonLineClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        node_id: str | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id
        self.timeout_seconds = timeout_seconds
        self.sock: socket.socket | None = None
        self._buffer = b""
        self._request_counter = 0

    def __enter__(self) -> "JsonLineClient":
        self.connect()
        self.handshake()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def connect(self) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_seconds)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except socket.timeout as exc:
            raise ProtocolTimeout(
                f"connect/handshake timed out after {self.timeout_seconds}s",
                field="socket",
                details={"host": self.host, "port": self.port},
            ) from exc
        except OSError as exc:
            raise RemoteProtocolError("E_CONNECT", f"cannot connect to {self.host}:{self.port}: {exc}") from exc

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _recv_line(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        if self.sock is None:
            raise FieldError("socket is not connected", field="socket")
        self.sock.settimeout(self.timeout_seconds if timeout_seconds is None else timeout_seconds)
        while b"\n" not in self._buffer:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout as exc:
                raise ProtocolTimeout(
                    f"socket read timed out after {self.sock.gettimeout()}s",
                    field="read_timeout",
                    details={"host": self.host, "port": self.port},
                ) from exc
            except OSError as exc:
                raise RemoteProtocolError("E_SOCKET", f"socket read failed: {exc}") from exc
            if not chunk:
                raise RemoteProtocolError("E_EOF", "node closed connection before a complete JSON line")
            self._buffer += chunk
        raw_line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            response = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FieldError("response is not a valid JSON object line", field="json_line") from exc
        if not isinstance(response, dict):
            raise FieldError("response must be a JSON object", field="$")
        return response

    def handshake(self) -> dict[str, Any]:
        response = self.request_raw(
            {"type": "hello", "protocol": PROTOCOL_VERSION, "node_id": self.node_id},
            validate=False,
        )
        for field in ("type", "protocol", "node_id"):
            if field not in response:
                raise FieldError("handshake response is missing field", entry="hello_ack", field=field)
        if response.get("type") != "hello_ack":
            raise FieldError("unexpected handshake response type", entry="hello", field="type")
        if response.get("node_id") != self.node_id:
            raise FieldError("handshake node_id does not match declaration", entry="hello_ack", field="node_id")
        return response

    def request_raw(self, payload: dict[str, Any], *, validate: bool = True) -> dict[str, Any]:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        if "request_id" not in payload:
            self._request_counter += 1
            payload["request_id"] = self._request_counter
        try:
            self.sock.sendall(encode_line(payload))
        except socket.timeout as exc:
            raise ProtocolTimeout("socket write timed out", field="write_timeout") from exc
        except OSError as exc:
            raise RemoteProtocolError("E_SOCKET", f"socket write failed: {exc}") from exc
        response = self._recv_line()
        if validate:
            self._validate_response(payload, response)
        return response

    def call(self, op: str, **fields: Any) -> dict[str, Any]:
        response = self.request_raw({"type": "request", "op": op, **fields})
        return response

    def _validate_response(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        if "ok" not in response or not isinstance(response.get("ok"), bool):
            raise FieldError("response is missing boolean ok field", entry=request.get("op"), field="ok")
        if "op" not in response:
            raise FieldError("response is missing op field", entry=request.get("op"), field="op")
        if response.get("op") != request.get("op"):
            raise FieldError(
                "response op does not match request",
                entry=request.get("op"),
                field="op",
                details={"response_op": response.get("op")},
            )
        if not response["ok"]:
            error = response.get("error")
            if not isinstance(error, dict) or "code" not in error:
                raise FieldError("error response is missing error.code", entry=request.get("op"), field="error.code")
            code = str(error["code"])
            if code == "E_STALE":
                current = error.get("current_generation")
                if not isinstance(current, int):
                    raise FieldError("E_STALE response is missing integer current_generation", field="error.current_generation")
                raise StaleGenerationError(
                    str(error.get("message", "stale generation")),
                    current_generation=current,
                    entry=request.get("lease_id"),
                    field="generation",
                )
            raise RemoteProtocolError(code, str(error.get("message", code)), error)

        for field in REQUIRED_RESPONSE_FIELDS.get(str(request.get("op")), ()):
            if field not in response:
                raise FieldError("success response is missing required field", entry=request.get("op"), field=field)


def wait_for_port(host: str, port: int, *, timeout_seconds: float = 4.0, node_id: str | None = None) -> JsonLineClient:
    deadline = time_monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time_monotonic() < deadline:
        client = JsonLineClient(host, port, node_id=node_id, timeout_seconds=min(0.5, timeout_seconds))
        try:
            client.connect()
            client.handshake()
            return client
        except (OSError, RemoteProtocolError, ProtocolTimeout, FieldError) as exc:
            last_error = exc
            client.close()
            sleep_small()
    raise ProtocolTimeout(
        f"node {node_id or host}:{port} did not complete handshake within {timeout_seconds}s",
        field="handshake",
        details={"last_error": repr(last_error)},
    )


def time_monotonic() -> float:
    import time

    return time.monotonic()


def sleep_small() -> None:
    import time

    time.sleep(0.03)

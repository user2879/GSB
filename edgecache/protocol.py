"""Loopback JSON-lines client used by the dispatcher."""

from __future__ import annotations

import json
import socket
import time
from typing import Any

from .errors import EdgeError, HandshakeTimeout, LeaseError, NodeError, PortError
from .model import Config, NodeSpec

_MAX_LINE = 16 * 1024 * 1024


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _decode_line(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NodeError(f"invalid JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise NodeError("protocol response must be an object")
    return value


class NodeClient:
    def __init__(self, node: NodeSpec, config: Config) -> None:
        self.node = node
        self.config = config

    def _read_line(self, sock: socket.socket, timeout_ms: int) -> dict[str, Any]:
        sock.settimeout(timeout_ms / 1000.0)
        chunks = bytearray()
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    raise NodeError("node closed the connection", node_id=self.node.node_id)
                chunks.extend(chunk)
                if len(chunks) > _MAX_LINE:
                    raise NodeError("response line exceeds size limit", node_id=self.node.node_id)
                newline = chunks.find(b"\n")
                if newline >= 0:
                    return _decode_line(bytes(chunks[:newline]))
        except socket.timeout as exc:
            raise HandshakeTimeout("node communication timed out", node_id=self.node.node_id) from exc

    def _write_line(self, sock: socket.socket, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sock.sendall(payload + b"\n")

    @staticmethod
    def _raise_response_error(response: dict[str, Any], node_id: str) -> None:
        if response.get("ok") is True:
            return
        code = str(response.get("code", "E_NODE"))
        message = str(response.get("message", "node rejected request"))
        if code == "E_LEASE":
            raise LeaseError(message, node_id=node_id)
        if code == "E_PORT":
            raise PortError(message, node_id=node_id)
        if code == "E_TIMEOUT":
            raise HandshakeTimeout(message, node_id=node_id)
        raise NodeError(message, node_id=node_id, code=code)

    def call(self, op: str, payload: dict[str, Any] | None = None, *, quota_version: int | None = None) -> dict[str, Any]:
        deadline_ms = self.config.handshake_timeout_ms
        try:
            with socket.create_connection(
                (self.node.host, self.node.port), timeout=self.config.socket_timeout_ms / 1000.0
            ) as sock:
                hello = {
                    "op": "hello",
                    "node_id": self.node.node_id,
                    "client": "edgecache-dispatcher/1",
                }
                if quota_version is not None:
                    hello["quota_version"] = quota_version
                self._write_line(sock, hello)
                hello_response = self._read_line(sock, deadline_ms)
                self._raise_response_error(hello_response, self.node.node_id)
                if hello_response.get("node_id") != self.node.node_id:
                    raise NodeError("handshake node_id mismatch", node_id=self.node.node_id)
                if op == "hello":
                    return hello_response
                request = dict(payload or {})
                request["op"] = op
                if quota_version is not None:
                    request["quota_version"] = quota_version
                self._write_line(sock, request)
                response = self._read_line(sock, self.config.socket_timeout_ms)
                self._raise_response_error(response, self.node.node_id)
                return response
        except TimeoutError as exc:
            raise HandshakeTimeout("connect timed out", node_id=self.node.node_id) from exc
        except socket.timeout as exc:
            raise HandshakeTimeout("socket timed out", node_id=self.node.node_id) from exc
        except (ConnectionError, OSError) as exc:
            raise NodeError(f"node socket failure: {exc}", node_id=self.node.node_id) from exc
        except EdgeError:
            raise

    def hello(self, quota_version: int) -> dict[str, Any]:
        return self.call("hello", quota_version=quota_version)

    def validate_batch(self, batch_id: str, items: list[dict[str, Any]], quota_version: int) -> dict[str, Any]:
        return self.call(
            "validate_batch",
            {"batch_id": batch_id, "items": items},
            quota_version=quota_version,
        )

    def renew(self, lease_id: str, expected_version: int, quota_version: int) -> dict[str, Any]:
        return self.call(
            "renew",
            {"lease_id": lease_id, "expected_version": expected_version},
            quota_version=quota_version,
        )

    def release(self, lease_id: str, quota_version: int) -> dict[str, Any]:
        return self.call("release", {"lease_id": lease_id}, quota_version=quota_version)

    def restore(self, leases: list[dict[str, Any]], quota_version: int) -> dict[str, Any]:
        return self.call("restore", {"leases": leases}, quota_version=quota_version)

    def set_quota(self, quota_version: int) -> dict[str, Any]:
        return self.call("set_quota", {"quota_version": quota_version})

    def shutdown(self, quota_version: int) -> dict[str, Any]:
        return self.call("shutdown", quota_version=quota_version)

"""JSON-lines loopback protocol and strict response validation."""
from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Any

from .errors import ArbiterError, HandshakeError, ProtocolError, ReplyError
from .util import canonical_bytes

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 4 * 1024 * 1024


def encode_line(value: dict[str, Any]) -> bytes:
    return canonical_bytes(value) + b"\n"


def read_line(sock: socket.socket, max_bytes: int = MAX_LINE_BYTES) -> dict[str, Any]:
    buffer = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ProtocolError("connection closed before a complete JSON line", field="newline")
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise ProtocolError("JSON line exceeds size limit", field="line_size", limit=max_bytes)
        newline = buffer.find(b"\n")
        if newline >= 0:
            raw = bytes(buffer[:newline])
            if not raw:
                raise ProtocolError("empty JSON line", field="line")
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError("invalid JSON line", field="json", error=str(exc)) from exc
            if not isinstance(decoded, dict):
                raise ProtocolError("protocol line must be a JSON object", field="root")
            return decoded


@dataclass
class NodeClient:
    host: str
    port: int
    timeout_s: float = 2.0

    def _round_trip(
        self,
        request: dict[str, Any],
        *,
        timeout_s: float | None = None,
        phase: str = "reply",
        node_id: str | None = None,
    ) -> dict[str, Any]:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(encode_line(request))
                response = read_line(sock)
        except socket.timeout as exc:
            if phase == "handshake":
                raise HandshakeError(
                    "handshake timed out",
                    node_id=node_id,
                    host=self.host,
                    port=self.port,
                ) from exc
            raise ReplyError(
                "node reply timed out",
                node_id=node_id,
                host=self.host,
                port=self.port,
            ) from exc
        except (ConnectionError, OSError) as exc:
            if phase == "handshake":
                raise HandshakeError(
                    "handshake connection failed",
                    node_id=node_id,
                    host=self.host,
                    port=self.port,
                    error=str(exc),
                ) from exc
            raise ReplyError(
                "node connection failed",
                node_id=node_id,
                host=self.host,
                port=self.port,
                error=str(exc),
            ) from exc
        if not isinstance(response, dict):
            raise ReplyError("reply is not a JSON object", node_id=node_id)
        if response.get("ok") is False:
            code = response.get("code", "E_REPLY")
            message = response.get("message", "node returned an error")
            details = {key: value for key, value in response.items() if key not in {"ok", "code", "message"}}
            details.setdefault("node_id", node_id)
            if code == "E_REPLY":
                raise ReplyError(message, **details)
            if code == "E_PROTOCOL":
                raise ProtocolError(message, **details)
            raise ArbiterError(message, remote_code=code, **details)
        return response

    @staticmethod
    def _require_fields(response: dict[str, Any], fields: tuple[str, ...], node_id: str | None) -> None:
        missing = [field for field in fields if field not in response]
        if missing:
            raise ReplyError(
                "node reply is missing required fields",
                node_id=node_id,
                fields=missing,
            )

    def hello(self, node_id: str | None = None) -> dict[str, Any]:
        response = self._round_trip(
            {"type": "hello", "protocol": PROTOCOL_VERSION},
            phase="handshake",
            node_id=node_id,
        )
        self._require_fields(response, ("type", "node_id", "protocol"), node_id)
        if response["protocol"] != PROTOCOL_VERSION:
            raise HandshakeError(
                "protocol version mismatch",
                node_id=node_id,
                expected=PROTOCOL_VERSION,
                actual=response.get("protocol"),
            )
        return response

    def inventory(self, node_id: str | None = None) -> dict[str, Any]:
        response = self._round_trip({"type": "inventory", "protocol": PROTOCOL_VERSION}, node_id=node_id)
        self._require_fields(response, ("type", "node_id", "digests", "reusable_keys"), node_id)
        if not isinstance(response["digests"], dict):
            raise ReplyError("inventory digests must be an object", node_id=node_id, field="digests")
        if not isinstance(response["reusable_keys"], list):
            raise ReplyError("inventory reusable_keys must be an array", node_id=node_id, field="reusable_keys")
        return response

    def lease(
        self,
        *,
        node_id: str,
        request_id: str,
        artifact_keys: list[str],
        lease_ms: int,
        sequence: int,
    ) -> dict[str, Any]:
        response = self._round_trip(
            {
                "type": "lease",
                "protocol": PROTOCOL_VERSION,
                "request_id": request_id,
                "sequence": sequence,
                "artifact_keys": artifact_keys,
                "lease_ms": lease_ms,
            },
            node_id=node_id,
        )
        self._require_fields(
            response,
            (
                "type",
                "lease_id",
                "node_id",
                "sequence",
                "granted_keys",
                "expires_at_ms",
                "digests",
                "reusable_keys",
            ),
            node_id,
        )
        if not isinstance(response["granted_keys"], list) or not response["granted_keys"]:
            raise ReplyError("lease reply granted_keys must be a non-empty array", node_id=node_id, field="granted_keys")
        if not isinstance(response["expires_at_ms"], int) or isinstance(response["expires_at_ms"], bool):
            raise ReplyError("lease reply expires_at_ms must be an integer", node_id=node_id, field="expires_at_ms")
        if not isinstance(response["digests"], dict):
            raise ReplyError("lease reply digests must be an object", node_id=node_id, field="digests")
        if response["sequence"] != sequence:
            raise ReplyError(
                "lease reply sequence does not match the request",
                node_id=node_id,
                field="sequence",
                expected=sequence,
                actual=response["sequence"],
            )
        if response["node_id"] != node_id:
            raise ReplyError(
                "lease reply node_id does not match the connected node",
                node_id=node_id,
                field="node_id",
                actual=response["node_id"],
            )
        if set(response["granted_keys"]) != set(artifact_keys):
            raise ReplyError(
                "lease reply did not grant the requested key set",
                node_id=node_id,
                field="granted_keys",
                expected=sorted(artifact_keys),
                actual=response["granted_keys"],
            )
        return response

    def renew(self, *, node_id: str, lease_id: str, extend_ms: int) -> dict[str, Any]:
        response = self._round_trip(
            {
                "type": "renew",
                "protocol": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "extend_ms": extend_ms,
            },
            node_id=node_id,
        )
        self._require_fields(response, ("type", "lease_id", "expires_at_ms"), node_id)
        return response

    def release(self, *, node_id: str, lease_id: str, reason: str = "client_release") -> dict[str, Any]:
        response = self._round_trip(
            {
                "type": "release",
                "protocol": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "reason": reason,
            },
            node_id=node_id,
        )
        self._require_fields(response, ("type", "lease_id", "released"), node_id)
        return response

    def status(self, node_id: str | None = None) -> dict[str, Any]:
        response = self._round_trip({"type": "status", "protocol": PROTOCOL_VERSION}, node_id=node_id)
        self._require_fields(
            response,
            ("type", "node_id", "capacity_slots", "capacity_used", "lease_limit", "active_lease_count"),
            node_id,
        )
        return response

    def shutdown(self, node_id: str | None = None) -> dict[str, Any]:
        try:
            return self._round_trip(
                {"type": "shutdown", "protocol": PROTOCOL_VERSION},
                timeout_s=1.0,
                node_id=node_id,
            )
        except (ReplyError, HandshakeError, ProtocolError):
            return {"ok": False, "type": "shutdown_ack", "node_id": node_id}


class FakeServer:
    """Tiny test helper for deterministic handshake/reply failure tests."""

    def __init__(self, mode: str = "silent") -> None:
        self.mode = mode
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> "FakeServer":
        self._thread.start()
        return self

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            with conn:
                if self.mode == "silent":
                    self._stop.wait(2.0)
                    continue
                try:
                    read_line(conn)
                    if self.mode == "missing":
                        conn.sendall(encode_line({"ok": True, "type": "hello", "protocol": PROTOCOL_VERSION, "node_id": "fake"}))
                        # Keep the connection open; the next line is a lease
                        # request and gets a response missing its lease_id.
                        read_line(conn)
                        conn.sendall(
                            encode_line(
                                {
                                    "ok": True,
                                    "type": "lease_granted",
                                    "node_id": "fake",
                                    "granted_keys": ["x"],
                                    "expires_at_ms": 1,
                                    "digests": {},
                                    "reusable_keys": [],
                                }
                            )
                        )
                    elif self.mode == "bad_json":
                        conn.sendall(b"not-json\n")
                except (OSError, Exception):
                    pass

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

"""Minimal newline-delimited JSON TCP protocol."""

from __future__ import annotations

import json
import socket
from typing import Any, Iterable

from .errors import DiagnosticError

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 32 * 1024 * 1024


def encode_line(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"


def send_json_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(encode_line(payload))


def recv_json_line(sock: socket.socket, timeout_s: float, *, peer: str = "worker") -> dict[str, Any]:
    if timeout_s <= 0:
        raise DiagnosticError("E_TIMEOUT", "socket timeout must be positive", field="timeout_s")
    chunks = bytearray()
    while True:
        if len(chunks) > MAX_LINE_BYTES:
            raise DiagnosticError(
                "E_PROTOCOL", f"{peer} response line exceeds {MAX_LINE_BYTES} bytes"
            )
        try:
            chunk = sock.recv(65536)
        except socket.timeout as exc:
            raise DiagnosticError(
                "E_TIMEOUT", f"timed out waiting for {peer} response", details={"timeout_s": timeout_s}
            ) from exc
        except OSError as exc:
            raise DiagnosticError("E_PROTOCOL", f"socket read failed: {exc}") from exc
        if not chunk:
            raise DiagnosticError("E_PROTOCOL", f"{peer} closed the connection before a line")
        chunks.extend(chunk)
        newline = chunks.find(b"\n")
        if newline >= 0:
            raw = bytes(chunks[:newline])
            break
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError("E_PROTOCOL", f"invalid JSON line from {peer}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError("E_PROTOCOL", f"{peer} response must be a JSON object")
    return value


class TcpShardClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 2.0,
        handshake_timeout_s: float = 3.0,
        read_timeout_s: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.handshake_timeout_s = handshake_timeout_s
        self.read_timeout_s = read_timeout_s
        self.pid: int | None = None

    def _call(
        self, request: dict[str, Any], required: Iterable[str], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        try:
            with socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s
            ) as sock:
                sock.settimeout(timeout_s if timeout_s is not None else self.read_timeout_s)
                send_json_line(sock, request)
                response = recv_json_line(
                    sock,
                    timeout_s if timeout_s is not None else self.read_timeout_s,
                    peer=f"worker {self.host}:{self.port}",
                )
        except DiagnosticError:
            raise
        except (socket.timeout, TimeoutError) as exc:
            raise DiagnosticError(
                "E_TIMEOUT",
                f"timed out communicating with worker {self.host}:{self.port}",
            ) from exc
        except ConnectionRefusedError as exc:
            raise DiagnosticError(
                "E_PORT", f"worker is not listening on {self.host}:{self.port}"
            ) from exc
        except OSError as exc:
            raise DiagnosticError(
                "E_PROTOCOL", f"worker socket failure on {self.host}:{self.port}: {exc}"
            ) from exc
        if response.get("ok") is False:
            raise DiagnosticError(
                str(response.get("code", "E_PROTOCOL")),
                str(response.get("message", "worker rejected request")),
                field=response.get("field"),
            )
        for field in required:
            if field not in response:
                raise DiagnosticError(
                    "E_FIELD",
                    f"worker response is missing required field {field!r}",
                    field=f"response.{field}",
                )
        return response

    def handshake(self, shard_id: str) -> dict[str, Any]:
        response = self._call(
            {
                "type": "handshake",
                "protocol_version": PROTOCOL_VERSION,
                "shard_id": shard_id,
            },
            ("ok", "shard_id", "pid", "protocol_version"),
            timeout_s=self.handshake_timeout_s,
        )
        if response["shard_id"] != shard_id:
            raise DiagnosticError(
                "E_FIELD",
                f"handshake shard_id mismatch: {response['shard_id']!r}",
                field="response.shard_id",
            )
        if response["protocol_version"] != PROTOCOL_VERSION:
            raise DiagnosticError(
                "E_PROTOCOL",
                f"unsupported worker protocol version {response['protocol_version']}",
            )
        self.pid = int(response["pid"])
        return response

    def run_batch(
        self,
        *,
        shard_id: str,
        batch_id: str,
        sample_ids: Iterable[int],
        parameter_version: int,
        checkpoint_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        request = {
            "type": "batch",
            "request_id": f"{shard_id}:{batch_id}:{parameter_version}",
            "shard_id": shard_id,
            "batch_id": batch_id,
            "sample_ids": list(sample_ids),
            "parameter_version": parameter_version,
        }
        if checkpoint_id is not None:
            request["checkpoint_id"] = checkpoint_id
        response = self._call(
            request,
            (
                "batch_id",
                "shard_id",
                "sample_ids",
                "parameter_version",
                "parameter_digest",
                "loss",
                "loss_digest",
                "elapsed_ms",
            ),
            timeout_s=timeout_s,
        )
        if response["batch_id"] != batch_id:
            raise DiagnosticError(
                "E_FIELD", "batch response batch_id mismatch", field="response.batch_id"
            )
        if response["shard_id"] != shard_id:
            raise DiagnosticError(
                "E_FIELD", "batch response shard_id mismatch", field="response.shard_id"
            )
        if response["parameter_version"] != parameter_version:
            raise DiagnosticError(
                "E_FIELD",
                "batch response parameter_version mismatch",
                field="response.parameter_version",
            )
        if checkpoint_id is not None:
            if response.get("checkpoint_id") != checkpoint_id:
                raise DiagnosticError(
                    "E_FIELD",
                    "batch response checkpoint_id mismatch",
                    field="response.checkpoint_id",
                )
        return response

    def lease_issue(self, shard_id: str, lease_id: str, ttl_ms: int) -> dict[str, Any]:
        return self._call(
            {
                "type": "lease_issue",
                "shard_id": shard_id,
                "lease_id": lease_id,
                "ttl_ms": ttl_ms,
            },
            ("ok", "lease_id", "lease_version", "expires_at_ms"),
        )

    def lease_renew(
        self, shard_id: str, lease_id: str, lease_version: int, ttl_ms: int
    ) -> dict[str, Any]:
        return self._call(
            {
                "type": "lease_renew",
                "shard_id": shard_id,
                "lease_id": lease_id,
                "lease_version": lease_version,
                "ttl_ms": ttl_ms,
            },
            ("ok", "lease_id", "lease_version", "expires_at_ms"),
        )

    def lease_release(self, shard_id: str, lease_id: str) -> dict[str, Any]:
        return self._call(
            {"type": "lease_release", "shard_id": shard_id, "lease_id": lease_id},
            ("ok", "lease_id", "state"),
        )

    def ping(self, shard_id: str) -> dict[str, Any]:
        return self._call({"type": "ping", "shard_id": shard_id}, ("ok", "shard_id", "pid"))

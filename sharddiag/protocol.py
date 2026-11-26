"""JSON Lines TCP protocol client and receipt validation."""

from __future__ import annotations

import json
import socket
from typing import Any, Iterable

from .errors import DiagnosticError, PortError, ProtocolError, TimeoutDiagnosticError


PROTOCOL_VERSION = 1


class JsonLineClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        expected_node_id: str | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        if host != "127.0.0.1":
            raise ProtocolError("protocol host must be literal 127.0.0.1", field="host")
        self.host = host
        self.port = int(port)
        self.expected_node_id = expected_node_id
        self.timeout_seconds = max(0.001, int(timeout_ms) / 1000.0)
        self.sock: socket.socket | None = None
        self._buffer = b""

    def connect(self) -> None:
        if self.sock is not None:
            return
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_seconds)
            self.sock.settimeout(self.timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutDiagnosticError(
                "TCP connect/handshake timed out", host=self.host, port=self.port
            ) from exc
        except socket.timeout as exc:
            raise TimeoutDiagnosticError(
                "TCP connect/handshake timed out", host=self.host, port=self.port
            ) from exc
        except OSError as exc:
            raise PortError(
                f"cannot connect to {self.host}:{self.port}: {exc}",
                host=self.host,
                port=self.port,
            ) from exc

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
                self._buffer = b""

    def __enter__(self) -> "JsonLineClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _read_line(self, timeout_ms: int | None = None) -> bytes:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        if timeout_ms is not None:
            self.sock.settimeout(max(0.001, int(timeout_ms) / 1000.0))
        while b"\n" not in self._buffer:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout as exc:
                raise TimeoutDiagnosticError(
                    "socket read timed out", host=self.host, port=self.port
                ) from exc
            except TimeoutError as exc:
                raise TimeoutDiagnosticError(
                    "socket read timed out", host=self.host, port=self.port
                ) from exc
            except OSError as exc:
                raise ProtocolError(
                    f"socket read failed: {exc}", host=self.host, port=self.port
                ) from exc
            if not chunk:
                raise ProtocolError(
                    "connection closed before a complete JSON line was received",
                    host=self.host,
                    port=self.port,
                )
            self._buffer += chunk
        line, remainder = self._buffer.split(b"\n", 1)
        self._buffer = remainder
        return line

    def request(
        self,
        payload: dict[str, Any],
        *,
        required_fields: Iterable[str] = (),
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        self.connect()
        assert self.sock is not None
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if timeout_ms is not None:
            self.sock.settimeout(max(0.001, int(timeout_ms) / 1000.0))
        try:
            self.sock.sendall(data)
        except socket.timeout as exc:
            raise TimeoutDiagnosticError("socket write timed out", host=self.host, port=self.port) from exc
        except TimeoutError as exc:
            raise TimeoutDiagnosticError("socket write timed out", host=self.host, port=self.port) from exc
        except OSError as exc:
            raise ProtocolError(f"socket write failed: {exc}", host=self.host, port=self.port) from exc

        line = self._read_line(timeout_ms)
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("receipt is not a valid JSON line", raw_prefix=line[:120].decode("utf-8", "replace")) from exc
        if not isinstance(response, dict):
            raise ProtocolError("receipt must be a JSON object")
        if response.get("ok") is False or response.get("error_code") or response.get("code"):
            code = str(response.get("error_code") or response.get("code") or "E_PROTO")
            if code not in {"E_PORT", "E_TIMEOUT", "E_PROTO", "E_LEASE"}:
                code = "E_PROTO"
            extra = {
                k: v
                for k, v in response.items()
                if k not in {"ok", "code", "error_code", "error", "message", "details"}
            }
            nested_details = response.get("details")
            if isinstance(nested_details, dict):
                extra.update(nested_details)
            raise DiagnosticError(
                code,
                str(response.get("error") or response.get("message") or "node request failed"),
                **extra,
            )
        missing = [name for name in required_fields if name not in response]
        if missing:
            raise ProtocolError(
                "receipt is missing required field(s)",
                missing_fields=missing,
                host=self.host,
                port=self.port,
            )
        return response

    def handshake(self, *, timeout_ms: int | None = None) -> dict[str, Any]:
        response = self.request(
            {"type": "hello", "protocol": PROTOCOL_VERSION},
            required_fields=("type", "protocol", "node_id"),
            timeout_ms=timeout_ms,
        )
        if response.get("type") != "hello_ack":
            raise ProtocolError("unexpected handshake response type", actual=response.get("type"))
        if response.get("protocol") != PROTOCOL_VERSION:
            raise ProtocolError(
                "unsupported protocol version",
                expected=PROTOCOL_VERSION,
                actual=response.get("protocol"),
            )
        if self.expected_node_id and response.get("node_id") != self.expected_node_id:
            raise ProtocolError(
                "handshake node_id does not match declaration",
                expected=self.expected_node_id,
                actual=response.get("node_id"),
            )
        return response

    def compute(self, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        return self.request(
            {"type": "compute", **payload},
            required_fields=(
                "request_id",
                "node_id",
                "shard_id",
                "capacity_version",
                "summary",
                "elapsed_monotonic_ms",
            ),
            timeout_ms=timeout_ms,
        )

    def lease_grant(self, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        return self.request(
            {"type": "lease_grant", **payload},
            required_fields=("lease_id", "lease_version", "status"),
            timeout_ms=timeout_ms,
        )

    def lease_renew(self, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        return self.request(
            {"type": "lease_renew", **payload},
            required_fields=("lease_id", "lease_version", "renew_count", "status", "expires_monotonic_ms"),
            timeout_ms=timeout_ms,
        )

    def lease_release(self, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        return self.request(
            {"type": "lease_release", **payload},
            required_fields=("lease_id", "status"),
            timeout_ms=timeout_ms,
        )

    def capacity_advance(self, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        return self.request(
            {"type": "capacity_advance", **payload},
            required_fields=("status", "current_capacity_version"),
            timeout_ms=timeout_ms,
        )


def validate_compute_receipt(receipt: dict[str, Any], *, expected_node_id: str, expected_shard_id: str) -> None:
    summary = receipt.get("summary")
    if not isinstance(summary, dict):
        raise ProtocolError("compute receipt summary must be an object", field="summary")
    missing_summary = [
        name
        for name in ("node_id", "shard_id", "sample_rows", "capacity_version", "participating_rows", "summary_digest")
        if name not in summary
    ]
    if missing_summary:
        raise ProtocolError("compute summary is missing required field(s)", missing_fields=missing_summary)
    if receipt.get("node_id") != expected_node_id or summary.get("node_id") != expected_node_id:
        raise ProtocolError("compute receipt node_id mismatch", expected=expected_node_id)
    if receipt.get("shard_id") != expected_shard_id or summary.get("shard_id") != expected_shard_id:
        raise ProtocolError("compute receipt shard_id mismatch", expected=expected_shard_id)
    if receipt.get("capacity_version") != summary.get("capacity_version"):
        raise ProtocolError("receipt and summary capacity_version differ")
    elapsed = receipt.get("elapsed_monotonic_ms")
    if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0:
        raise ProtocolError("elapsed_monotonic_ms must be a non-negative integer")

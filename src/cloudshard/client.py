from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .aggregate import validate_receipt
from .errors import CloudShardError, FieldError, LeaseError, PortInUseError, ProtocolTimeoutError
from .spec import NodeSpec


def _read_json_line(reader: Any, timeout_s: float) -> dict[str, Any]:
    try:
        line = reader.readline()
    except (TimeoutError, socket.timeout) as exc:
        raise ProtocolTimeoutError(f"socket read timed out after {timeout_s:.3f}s", field="readline") from exc
    except OSError as exc:
        raise PortInUseError(f"loopback socket I/O error: {exc}", field="port") from exc
    if not line:
        raise FieldError("peer closed connection before a JSON line was received", field="response")
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FieldError(f"peer returned invalid JSON line: {exc}", field="response") from exc
    if not isinstance(payload, dict):
        raise FieldError("peer response must be a JSON object", field="response")
    if payload.get("type") == "error" or payload.get("ok") is False:
        code = payload.get("code", "E_FIELD")
        message = str(payload.get("error", payload.get("message", "node returned an error")))
        if code == "E_LEASE":
            raise LeaseError(message, field=payload.get("field"), basis=payload.get("basis"))
        if code == "E_TIMEOUT":
            raise ProtocolTimeoutError(message, field=payload.get("field"))
        if code == "E_PORT":
            raise PortInUseError(message, field=payload.get("field"))
        if code == "E_FIELD":
            raise FieldError(message, field=payload.get("field"))
        raise CloudShardError(message)
    return payload


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _connect(node: NodeSpec, timeout_ms: int) -> socket.socket:
    timeout_s = max(0.05, timeout_ms / 1000.0)
    try:
        sock = socket.create_connection((node.host, node.port), timeout=timeout_s)
        sock.settimeout(timeout_s)
        return sock
    except (TimeoutError, socket.timeout) as exc:
        raise ProtocolTimeoutError(f"connect to {node.host}:{node.port} timed out", field="port") from exc
    except ConnectionRefusedError as exc:
        raise PortInUseError(f"nothing accepting on {node.host}:{node.port} (E_PORT)", field="port") from exc
    except OSError as exc:
        raise PortInUseError(f"cannot connect to {node.host}:{node.port}: {exc}", field="port") from exc


def call_lease(node: NodeSpec, message: dict[str, Any], *, timeout_ms: int = 2_000) -> dict[str, Any]:
    with _connect(node, timeout_ms) as sock:
        reader = sock.makefile("rb")
        _send(sock, {"type": "hello", "protocol": 1, "client": "cloudshard"})
        ack = _read_json_line(reader, timeout_ms / 1000.0)
        if ack.get("type") != "hello_ack" or ack.get("node_id") != node.node_id:
            raise FieldError("invalid handshake acknowledgement", field="hello_ack")
        _send(sock, message)
        return _read_json_line(reader, timeout_ms / 1000.0)


def grant_lease(node: NodeSpec, lease: dict[str, Any], *, timeout_ms: int = 2_000) -> dict[str, Any]:
    return call_lease(node, {"type": "lease_grant", **lease}, timeout_ms=timeout_ms)


def renew_lease(
    node: NodeSpec,
    *,
    lease_id: str,
    expected_version: int,
    new_expires_at_ms: int,
    timeout_ms: int = 2_000,
) -> dict[str, Any]:
    return call_lease(
        node,
        {
            "type": "lease_renew",
            "lease_id": lease_id,
            "expected_version": expected_version,
            "new_expires_at_ms": new_expires_at_ms,
        },
        timeout_ms=timeout_ms,
    )


def release_lease(node: NodeSpec, lease_id: str, *, timeout_ms: int = 2_000) -> dict[str, Any]:
    return call_lease(node, {"type": "lease_release", "lease_id": lease_id}, timeout_ms=timeout_ms)


def dispatch_batch(
    node: NodeSpec,
    *,
    shard_id: str,
    revision: int,
    samples: dict[str, Any],
    config: dict[str, Any] | None = None,
    request_id: str | None = None,
    timeout_ms: int = 2_000,
) -> dict[str, Any]:
    timeout_s = max(0.05, timeout_ms / 1000.0)
    with _connect(node, timeout_ms) as sock:
        reader = sock.makefile("rb")
        try:
            _send(sock, {"type": "hello", "protocol": 1, "node_id": node.node_id})
            ack = _read_json_line(reader, timeout_s)
            if ack.get("type") != "hello_ack":
                raise FieldError("handshake response must be hello_ack", field="type")
            _send(
                sock,
                {
                    "type": "compute",
                    "request_id": request_id or f"{shard_id}-{revision}",
                    "shard_id": shard_id,
                    "revision": revision,
                    "config": config or {},
                    "samples": samples,
                },
            )
            receipt = _read_json_line(reader, timeout_s)
        except (TimeoutError, socket.timeout) as exc:
            raise ProtocolTimeoutError(f"compute read timed out after {timeout_s:.3f}s", field="readline") from exc
    validate_receipt(receipt, expected_node=node.node_id, expected_shard=shard_id, expected_revision=revision)
    return receipt


def handshake_only(host: str, port: int, *, timeout_ms: int = 300) -> dict[str, Any]:
    node = NodeSpec(
        node_id="probe",
        host=host,
        port=port,
        concurrency_quota=1,
        lease_limit=1,
        seed=0,
        capacity_version=1,
    )
    with _connect(node, timeout_ms) as sock:
        reader = sock.makefile("rb")
        _send(sock, {"type": "hello", "protocol": 1})
        return _read_json_line(reader, timeout_ms / 1000.0)


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def read_spec_path_hint() -> Path:
    return Path.cwd()

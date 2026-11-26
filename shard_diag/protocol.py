from __future__ import annotations

import json
import socket
from typing import Any, Iterable

from . import PROTOCOL_VERSION
from .errors import FieldError, LeaseError, PortError, ProtocolTimeout


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ensure_port_available(port: int, object_id: str | None = None) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(port)))
    except OSError as exc:
        raise PortError(
            f"127.0.0.1:{port} is occupied or cannot be bound: {exc}",
            field="nodes.port",
            object_id=object_id,
            details={"port": port, "errno": getattr(exc, "errno", None)},
        ) from exc
    finally:
        sock.close()


def check_spec_ports(nodes: Iterable[Any]) -> None:
    for node in nodes:
        ensure_port_available(node.port, node.node_id)


def _read_line(stream: Any) -> bytes:
    try:
        line = stream.readline()
    except socket.timeout as exc:
        raise ProtocolTimeout("timed out while reading JSON line", field="socket.read") from exc
    except OSError as exc:
        raise PortError(f"loopback read failed: {exc}", field="socket.read") from exc
    if not line:
        raise FieldError("peer closed connection before a complete JSON line", field="response")
    return line


def _write_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    try:
        sock.sendall((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    except socket.timeout as exc:
        raise ProtocolTimeout("timed out while writing JSON line", field="socket.write") from exc
    except OSError as exc:
        raise PortError(f"loopback socket write failed: {exc}", field="socket.write") from exc


def _decode_line(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FieldError(f"response is not a valid JSON object line: {exc}", field="response") from exc
    if not isinstance(value, dict):
        raise FieldError("response must be a JSON object", field="response")
    return value


def require_fields(payload: dict[str, Any], fields: Iterable[str], object_id: str | None = None) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise FieldError(
            "response is missing required field(s): " + ", ".join(missing),
            field=missing[0],
            object_id=object_id,
            details={"missing": missing},
        )


def request_json(
    port: int,
    payload: dict[str, Any],
    *,
    timeout_ms: int = 2_000,
    host: str = "127.0.0.1",
) -> dict[str, Any]:
    timeout = max(0.001, timeout_ms / 1000.0)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except socket.timeout as exc:
        raise ProtocolTimeout(f"connect to {host}:{port} timed out", field="socket.connect") from exc
    except OSError as exc:
        raise PortError(f"cannot connect to {host}:{port}: {exc}", field="socket.connect", details={"port": port}) from exc
    try:
        sock.settimeout(timeout)
        stream = sock.makefile("rb")
        hello = {"type": "hello", "protocol_version": PROTOCOL_VERSION, "client": "shard-diag"}
        _write_line(sock, hello)
        hello_ack = _decode_line(_read_line(stream))
        require_fields(hello_ack, ["type", "node_id", "protocol_version"])
        if hello_ack["type"] != "hello_ack":
            raise FieldError("first response must be hello_ack", field="response.type")
        _write_line(sock, payload)
        response = _decode_line(_read_line(stream))
        if response.get("ok") is False:
            code = response.get("code", "E_INTERNAL")
            message = response.get("message", "node rejected request")
            if code == "E_LEASE":
                raise LeaseError(message, field=response.get("field"), object_id=response.get("object_id"), details=response.get("details"))
            if code == "E_FIELD":
                raise FieldError(message, field=response.get("field"), object_id=response.get("object_id"), details=response.get("details"))
            if code == "E_TIMEOUT":
                raise ProtocolTimeout(message, field=response.get("field"), details=response.get("details"))
            from .errors import DiagnosticError

            raise DiagnosticError(message, code=code, field=response.get("field"))
        require_fields(response, ["ok"])
        return response
    finally:
        try:
            stream.close()
        except Exception:
            pass
        try:
            sock.close()
        except OSError:
            pass

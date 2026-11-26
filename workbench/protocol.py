from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Any

from .errors import FieldError, LeaseError, NodeGoneError, PortError, TimeoutError


def new_request_id(prefix: str = "req") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _timeout_seconds(timeout_ms: int) -> float:
    return max(0.001, float(timeout_ms) / 1000.0)


def connect_socket(host: str, port: int, timeout_ms: int) -> socket.socket:
    timeout = _timeout_seconds(timeout_ms)
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except socket.timeout as exc:
        raise TimeoutError(
            f"connect timeout after {timeout_ms}ms",
            host=host,
            port=port,
            stage="connect",
        ) from exc
    except ConnectionRefusedError as exc:
        raise NodeGoneError(
            "connection refused",
            host=host,
            port=port,
            stage="connect",
        ) from exc
    except OSError as exc:
        raise PortError(
            f"cannot connect to loopback port {port}: {exc}",
            host=host,
            port=port,
            stage="connect",
        ) from exc
    sock.settimeout(timeout)
    return sock


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    try:
        sock.sendall(data)
    except OSError as exc:
        raise NodeGoneError(f"send failed: {exc}") from exc


def read_message(sock: socket.socket, timeout_ms: int) -> dict[str, Any]:
    sock.settimeout(_timeout_seconds(timeout_ms))
    reader = sock.makefile("rb")
    try:
        line = reader.readline()
    except socket.timeout as exc:
        raise TimeoutError(
            f"read timeout after {timeout_ms}ms", stage="read"
        ) from exc
    except OSError as exc:
        raise NodeGoneError(f"read failed: {exc}", stage="read") from exc
    finally:
        try:
            reader.close()
        except OSError:
            pass
    if not line:
        raise NodeGoneError("node closed connection without a JSON line", stage="read")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FieldError(f"response is not valid JSON line: {exc}", field="$") from exc
    if not isinstance(message, dict):
        raise FieldError("response must be a JSON object", field="$")
    return message


def handshake(
    host: str,
    port: int,
    node_id: str,
    timeout_ms: int,
    *,
    sock: socket.socket | None = None,
) -> dict[str, Any]:
    owns_socket = sock is None
    sock = sock or connect_socket(host, port, timeout_ms)
    try:
        send_message(
            sock,
            {
                "type": "hello",
                "protocol_version": 1,
                "node_id": node_id,
            },
        )
        ack = read_message(sock, timeout_ms)
        if ack.get("type") != "hello_ack":
            raise FieldError("handshake missing hello_ack", field="type", response=ack)
        if ack.get("node_id") != node_id:
            raise FieldError(
                "handshake node_id mismatch",
                field="node_id",
                expected=node_id,
                actual=ack.get("node_id"),
            )
        return ack
    finally:
        if owns_socket:
            try:
                sock.close()
            except OSError:
                pass


def require_fields(message: dict[str, Any], fields: list[str], prefix: str = "") -> None:
    for field in fields:
        current: Any = message
        full = prefix + field
        parts = field.split(".")
        ok = True
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if not ok or current is None:
            raise FieldError("response field is missing", field=full, response=message)


def call_node(
    node: dict[str, Any],
    message: dict[str, Any],
    *,
    timeout_ms: int,
    handshake_first: bool = True,
    expect_summary: bool = True,
) -> dict[str, Any]:
    host = str(node["host"])
    port = int(node["port"])
    node_id = str(node["node_id"])
    sock = connect_socket(host, port, timeout_ms)
    try:
        if handshake_first:
            handshake(host, port, node_id, timeout_ms, sock=sock)
        send_message(sock, message)
        response = read_message(sock, timeout_ms)
        if response.get("ok") is False:
            code = response.get("code", "E_NODE")
            if code == "E_LEASE":
                raise LeaseError(
                    str(response.get("message", "lease optimistic check failed")),
                    **{k: v for k, v in response.items() if k not in {"ok", "code", "message"}},
                )
            if code == "E_FIELD":
                raise FieldError(
                    str(response.get("message", "node reported missing field")),
                    field=str(response.get("field", "$")),
                )
            raise NodeGoneError(
                str(response.get("message", "node returned failure")), code=code
            )
        required = ["ok", "request_id", "node_id"]
        if expect_summary:
            required.extend(
                [
                    "shard_id",
                    "capacity_version",
                    "elapsed_ms_monotonic",
                    "summary",
                ]
            )
        require_fields(response, required)
        if expect_summary:
            require_fields(
                response["summary"],
                ["kind", "participating_rows", "groups"],
                prefix="summary.",
            )
        if response["node_id"] != node_id:
            raise FieldError("node_id mismatch in receipt", field="node_id")
        return response
    finally:
        try:
            sock.close()
        except OSError:
            pass


def wait_for_handshake(
    node: dict[str, Any],
    *,
    startup_timeout_ms: int,
    socket_timeout_ms: int,
    process: Any | None = None,
) -> None:
    """Poll a child's real loopback port until handshake succeeds."""

    import time

    deadline = time.monotonic() + startup_timeout_ms / 1000.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            stderr_tail = read_process_log_tail(process)
            raise PortError(
                f"node process exited before handshake (rc={process.returncode}); "
                f"port {node['port']} may be occupied",
                node_id=node["node_id"],
                port=node["port"],
                returncode=process.returncode,
                stderr_tail=stderr_tail,
            )
        try:
            handshake(
                str(node["host"]),
                int(node["port"]),
                str(node["node_id"]),
                min(socket_timeout_ms, max(50, int((deadline - time.monotonic()) * 1000))),
            )
            return
        except (NodeGoneError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.03)
    if process is not None and process.poll() is not None:
        raise PortError(
            "node process exited and port is unavailable",
            node_id=node["node_id"],
            port=node["port"],
            returncode=process.returncode,
        )
    raise TimeoutError(
        f"handshake startup timed out after {startup_timeout_ms}ms",
        node_id=node["node_id"],
        port=node["port"],
        last_error=str(last_error),
    )


def read_process_log_tail(process: Any, max_bytes: int = 4000) -> str:
    stderr = getattr(process, "stderr", None)
    if stderr is None:
        log_path = getattr(process, "stderr_log_path", None)
        if log_path:
            try:
                data = Path(log_path).read_bytes()[-max_bytes:]
                return data.decode("utf-8", errors="replace")
            except OSError:
                return ""
        return ""
    try:
        return stderr.read()[-max_bytes:].decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort diagnostics
        return ""

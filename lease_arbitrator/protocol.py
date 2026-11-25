from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from .errors import FieldError, HandshakeError, PortError


MAX_LINE_BYTES = 8 * 1024 * 1024


def monotonic_ns() -> int:
    return time.monotonic_ns()


def elapsed_ms(start_ns: int) -> int:
    return max(0, (time.monotonic_ns() - start_ns) // 1_000_000)


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def port_is_open(host: str, port: int, timeout_seconds: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _recv_line(sock: socket.socket, max_bytes: int = MAX_LINE_BYTES) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("node closed connection before a complete JSON line")
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            break
        if total > max_bytes:
            raise FieldError("JSON line exceeds maximum size", field="line")
    return b"".join(chunks).split(b"\n", 1)[0]


def request_json_line(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout_ms: int = 3000,
) -> dict[str, Any]:
    """Send one JSON-line request and parse one JSON-line receipt."""
    timeout_seconds = max(0.001, timeout_ms / 1000.0)
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
            sock.settimeout(timeout_seconds)
            sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            line = _recv_line(sock)
    except (TimeoutError, socket.timeout) as exc:
        raise HandshakeError(
            f"socket timeout talking to {host}:{port}",
            entry_id=f"{host}:{port}",
            field="socket",
        ) from exc
    except OSError as exc:
        raise HandshakeError(
            f"socket error talking to {host}:{port}: {exc}",
            entry_id=f"{host}:{port}",
            field="socket",
        ) from exc

    try:
        response = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FieldError("node returned a non-JSON receipt line", entry_id=f"{host}:{port}", field="receipt") from exc
    if not isinstance(response, dict):
        raise FieldError("node receipt must be a JSON object", entry_id=f"{host}:{port}", field="receipt")
    if response.get("ok") is False:
        error = response.get("error") if isinstance(response.get("error"), dict) else {}
        code = str(error.get("code") or response.get("error_code") or "E_NODE")
        message = str(error.get("message") or response.get("message") or "node rejected request")
        if code == "E_LEASE":
            from .errors import LeaseConflictError

            raise LeaseConflictError(message, entry_id=response.get("node_id"), field="lease_version", details=response)
        raise ArbitrationNodeError(code, message, details=response)
    return response


class ArbitrationNodeError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def require_receipt_fields(
    response: dict[str, Any],
    fields: tuple[str, ...],
    *,
    node_id: str | None = None,
) -> None:
    for field_name in fields:
        if field_name not in response or response[field_name] is None:
            raise FieldError(
                f"node receipt is missing required field {field_name!r}",
                entry_id=node_id or response.get("node_id"),
                field=field_name,
                details={"receipt": response},
            )


def read_log_tail(log_path: Path | None, max_chars: int = 4000) -> str:
    if log_path is None or not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _looks_like_port_failure(text: str) -> bool:
    upper = text.upper()
    markers = (
        "E_PORT",
        "EADDRINUSE",
        "ADDRESS ALREADY IN USE",
        "ONLY ONE USAGE OF EACH SOCKET ADDRESS",
        "WINERROR 10048",
        "ERROR 10048",
    )
    return any(marker in upper for marker in markers)


def wait_for_node(
    *,
    host: str,
    port: int,
    node_id: str,
    process: subprocess.Popen[Any] | None,
    handshake_timeout_ms: int,
    request_timeout_ms: int,
    log_path: Path | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.05, handshake_timeout_ms / 1000.0)
    last_error = "no connection attempt succeeded"

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            tail = read_log_tail(log_path)
            if _looks_like_port_failure(tail):
                raise PortError(
                    f"node {node_id} could not bind {host}:{port} (port occupied)",
                    entry_id=node_id,
                    field="port",
                    details={"returncode": process.returncode, "log_tail": tail},
                )
            raise HandshakeError(
                f"node {node_id} exited before handshake",
                entry_id=node_id,
                field="handshake",
                details={"returncode": process.returncode, "log_tail": tail},
            )

        try:
            response = request_json_line(
                host,
                port,
                {"action": "ping", "request_seq": 0},
                timeout_ms=min(500, request_timeout_ms),
            )
            require_receipt_fields(response, ("node_id", "quota_version"), node_id=node_id)
            if response["node_id"] != node_id:
                raise FieldError(
                    "handshake node_id does not match declaration",
                    entry_id=node_id,
                    field="node_id",
                    details={"received": response["node_id"]},
                )
            return response
        except FieldError:
            raise
        except (HandshakeError, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.03)

    tail = read_log_tail(log_path)
    if _looks_like_port_failure(tail):
        raise PortError(
            f"node {node_id} port {port} is occupied",
            entry_id=node_id,
            field="port",
            details={"log_tail": tail},
        )
    raise HandshakeError(
        f"handshake timed out for node {node_id} at {host}:{port}: {last_error}",
        entry_id=node_id,
        field="handshake",
        details={"log_tail": tail},
    )

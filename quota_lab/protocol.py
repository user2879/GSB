from __future__ import annotations

import json
import socket
from typing import Any

from .errors import LeaseError, PortError, QuotaLabError, TimeoutError as LabTimeoutError


def request_jsonline(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout_ms: int = 1000,
) -> dict[str, Any]:
    timeout_seconds = max(timeout_ms, 1) / 1000.0
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
            sock.settimeout(timeout_seconds)
            sock.sendall((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
            buffer = bytearray()
            while b"\n" not in buffer:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionResetError("node closed connection before newline response")
                buffer.extend(chunk)
                if len(buffer) > 16 * 1024 * 1024:
                    raise QuotaLabError("response exceeded 16 MiB", code="E_PROTOCOL")
    except (socket.timeout, TimeoutError) as exc:
        raise LabTimeoutError(
            "socket handshake/read timed out",
            field=f"{host}:{port}",
            details={"timeout_ms": timeout_ms, "op": payload.get("op")},
        ) from exc
    except ConnectionRefusedError as exc:
        raise PortError(
            "loopback TCP connection refused; port is unavailable or node exited",
            field=f"{host}:{port}",
            details={"op": payload.get("op")},
        ) from exc
    except OSError as exc:
        raise PortError(
            "loopback TCP socket error",
            field=f"{host}:{port}",
            details={"errno": getattr(exc, "errno", None), "os_error": str(exc)},
        ) from exc

    line = bytes(buffer).split(b"\n", 1)[0]
    try:
        response = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuotaLabError("node returned invalid JSON line", code="E_PROTOCOL") from exc
    if not isinstance(response, dict):
        raise QuotaLabError("node response root must be object", code="E_PROTOCOL")
    if response.get("ok") is False:
        error = response.get("error") or {}
        code = error.get("code", "E_NODE") if isinstance(error, dict) else "E_NODE"
        message = error.get("message", "node rejected request") if isinstance(error, dict) else "node rejected request"
        if code == "E_LEASE":
            raise LeaseError(str(message), details=error.get("details") if isinstance(error, dict) else None)
        if code == "E_TIMEOUT":
            raise LabTimeoutError(str(message), details=error.get("details") if isinstance(error, dict) else None)
        if code == "E_PORT":
            raise PortError(str(message), details=error.get("details") if isinstance(error, dict) else None)
        raise QuotaLabError(str(message), code=str(code), details=error if isinstance(error, dict) else None)
    return response

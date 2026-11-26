from __future__ import annotations

import json
import socket
from typing import Any

from .errors import LeaseConflictError, PortInUseError, RpcTimeoutError, WaterLeaseError


PROTOCOL_NAME = "waterlease-jsonl"
PROTOCOL_VERSION = 1


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def read_message(reader: Any) -> dict[str, Any]:
    line = reader.readline()
    if not line:
        raise WaterLeaseError("peer closed connection without a response", code_override="E_EOF")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WaterLeaseError(f"invalid JSON line: {exc}", code_override="E_PROTOCOL") from exc
    if not isinstance(message, dict):
        raise WaterLeaseError("response must be a JSON object", code_override="E_PROTOCOL")
    return message


def raise_for_error(response: dict[str, Any]) -> None:
    if response.get("ok") is True:
        return
    error = response.get("error") if isinstance(response.get("error"), dict) else {}
    code = error.get("code", "E_RPC")
    message = str(error.get("message", "remote node returned an error"))
    path = str(error.get("path", "$"))
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    if code == "E_LEASE":
        raise LeaseConflictError(message, path=path, details=details)
    if code == "E_TIMEOUT":
        raise RpcTimeoutError(message, path=path, details=details)
    if code == "E_PORT":
        raise PortInUseError(message, path=path, details=details)
    raise WaterLeaseError(message, path=path, details={"code": code, **details})


def rpc(
    host: str,
    port: int,
    message: dict[str, Any],
    *,
    timeout_ms: int = 2000,
    node_id: str | None = None,
) -> dict[str, Any]:
    timeout = max(0.001, timeout_ms / 1000.0)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            reader = sock.makefile("rb")
            hello = {
                "type": "hello",
                "protocol": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "node_id": node_id,
            }
            send_message(sock, hello)
            ack = read_message(reader)
            raise_for_error(ack)
            if ack.get("type") != "hello_ack":
                raise WaterLeaseError(
                    "invalid handshake acknowledgement",
                    code_override="E_PROTOCOL",
                    details={"ack": ack},
                )

            send_message(sock, message)
            response = read_message(reader)
            raise_for_error(response)
            return response
    except RpcTimeoutError:
        raise
    except (TimeoutError, socket.timeout) as exc:
        raise RpcTimeoutError(
            f"TCP RPC timed out after {timeout_ms} ms",
            path=f"$.port[{port}]",
            details={"host": host, "port": port},
        ) from exc
    except ConnectionRefusedError as exc:
        raise PortInUseError(
            "node port is not accepting JSON-lines connections",
            path=f"$.port[{port}]",
            details={"host": host, "port": port, "reason": "connection_refused"},
        ) from exc
    except OSError as exc:
        raise WaterLeaseError(
            f"TCP RPC failed: {exc}",
            code_override="E_SOCKET",
            details={"host": host, "port": port},
        ) from exc

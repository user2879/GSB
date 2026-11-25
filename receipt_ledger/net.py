from __future__ import annotations

import json
import socket
from typing import Any


def call_worker(
    port: int,
    payload: dict[str, Any],
    host: str = "127.0.0.1",
    timeout: float = 10.0,
) -> dict[str, Any]:
    request = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(request)
        stream = sock.makefile("rb")
        line = stream.readline()
    if not line:
        raise ConnectionError(f"工作进程 {host}:{port} 未返回响应")
    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "worker request failed"))
    return response


def port_is_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

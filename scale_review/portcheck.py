from __future__ import annotations

import socket
from typing import Any


def check_loopback_port(host: str = "127.0.0.1", port: int = 0) -> dict[str, Any]:
    """Bind a loopback socket to prove the local port stack is usable."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        bound_port = sock.getsockname()[1]
        return {
            "ok": True,
            "host": host,
            "requested_port": port,
            "available_port": bound_port,
            "ephemeral": port == 0,
        }

"""JSON-lines over loopback TCP message helpers."""
from __future__ import annotations

import json
import socket
from typing import Any

HOST = "127.0.0.1"


def send_message(sock, message: dict) -> None:
    payload = (json.dumps(message, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
    if hasattr(sock, "sendall"):
        sock.sendall(payload)
    else:  # socketserver StreamRequestHandler.wfile file-like object
        sock.write(payload)
        sock.flush()


def recv_message(sock: socket.socket) -> dict:
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("peer closed connection before newline")
        buffer.extend(chunk)
    return json.loads(buffer.decode("utf-8"))


def request(host: str, port: int, message: dict, timeout: float = 10.0) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_message(sock, message)
        return recv_message(sock)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]

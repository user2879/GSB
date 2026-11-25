"""Synchronous JSON-lines client for node worker subprocesses."""

import json
import socket


class NodeClient:
    def __init__(self, host, port, timeout_ms=10_000):
        self.host = host
        self.port = port
        self.timeout = timeout_ms / 1000.0

    def request(self, payload: dict) -> dict:
        line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as conn:
            conn.sendall(line)
            buffer = b""
            while b"\n" not in buffer:
                chunk = conn.recv(65536)
                if not chunk:
                    raise ConnectionError(f"node at {self.host}:{self.port} closed without response")
                buffer += chunk
        return json.loads(buffer.split(b"\n", 1)[0].decode("utf-8"))

    def ping(self) -> dict:
        return self.request({"type": "ping"})

    def shutdown(self) -> dict:
        return self.request({"type": "shutdown"})

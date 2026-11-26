"""JSON-line protocol used over loopback TCP.

Each request is one JSON object terminated by ``\\n``; each response is one
JSON object terminated by ``\\n``.  Message types:

* ``acquire``  - create a lease, response carries the first generation tag
* ``renew``    - requires the current tag; success advances the generation
* ``release``  - requires the current tag; success advances then frees the slot
* ``compute``  - requires the current tag; runs the NumPy verification
* ``restore``  - administrative rebuild of node lease algebra state
* ``ping``     - liveness probe
* ``shutdown`` - graceful node exit

The client opens one short-lived connection per request.  Concurrent centre
threads therefore never share a TCP stream and cannot interleave JSON lines.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any

from .errors import E_PROTOCOL, EdgeCacheError, Failure

ACQUIRE = "acquire"
RENEW = "renew"
RELEASE = "release"
COMPUTE = "compute"
RESTORE = "restore"
PING = "ping"
SHUTDOWN = "shutdown"

DEFAULT_TIMEOUT_S = 10.0


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def decode_message(line: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EdgeCacheError(
            Failure(E_PROTOCOL, "malformed JSON line", {"error": str(exc)})
        ) from exc
    if not isinstance(payload, dict):
        raise EdgeCacheError(Failure(E_PROTOCOL, "message must be a JSON object"))
    return payload


class NodeClient:
    """Blocking JSON-line client; one connection per request."""

    def __init__(self, port: int, host: str = "127.0.0.1", timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def _connect(self, retries: int = 50, retry_delay_s: float = 0.05) -> socket.socket:
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                sock.settimeout(self.timeout_s)
                return sock
            except OSError as exc:
                last_error = exc
                time.sleep(retry_delay_s)
        raise EdgeCacheError(
            Failure(E_PROTOCOL, "cannot connect to node",
                    {"port": self.port, "error": str(last_error)})
        )

    def connect(self) -> None:
        # Retained for lifecycle callers (pool readiness); the socket itself
        # is not kept because requests must not share a stream.
        with self._connect():
            pass

    def close(self) -> None:
        return None

    def __enter__(self) -> "NodeClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._connect() as sock:
                sock.sendall(encode_message(message))
                buffer = bytearray()
                while not buffer.endswith(b"\n"):
                    chunk = sock.recv(65536)
                    if not chunk:
                        raise EdgeCacheError(Failure(E_PROTOCOL, "node closed connection"))
                    buffer.extend(chunk)
                return decode_message(bytes(buffer))
        except EdgeCacheError:
            raise
        except OSError as exc:
            raise EdgeCacheError(
                Failure(E_PROTOCOL, "protocol exchange failed", {"error": str(exc)})
            ) from exc

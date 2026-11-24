"""Real loopback socket transport between scheduler and local workers.

Frames are length-prefixed JSON: a 4-byte big-endian unsigned length followed
by UTF-8 JSON bytes. The client enforces a real per-attempt ``socket.settimeout``
deadline and retries idempotent batch requests; retries never change the
single-attempt semantics because the worker deterministically recomputes the
same response for the same batch payload.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from dataclasses import dataclass

from .errors import TimeoutError_

LENGTH_HEADER = struct.Struct(">I")
DEFAULT_TIMEOUT_MS = 1500


def _send_frame(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sock.sendall(LENGTH_HEADER.pack(len(body)) + body)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed before full frame was received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(sock: socket.socket) -> dict:
    (length,) = LENGTH_HEADER.unpack(_recv_exact(sock, LENGTH_HEADER.size))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def check_port_bindable(host: str, port: int) -> None:
    """Raise OSError if the loopback port cannot be bound right now."""

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    finally:
        probe.close()


@dataclass
class WorkerService:
    """Deterministic loopback worker serving forward-pass requests."""

    host: str
    port: int
    compute_callback: callable
    server: socket.socket | None = None
    thread: threading.Thread | None = None
    _stop: threading.Event | None = None

    def start(self) -> None:
        check_port_bindable(self.host, self.port)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(16)
        self.server.settimeout(0.1)
        self._stop = threading.Event()
        self.thread = threading.Thread(
            target=self._serve_loop, name=f"worker-{self.port}", daemon=True
        )
        self.thread.start()

    def _serve_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn,), name=f"conn-{self.port}", daemon=True
            ).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                request = _recv_frame(conn)
                response = self.compute_callback(request)
                _send_frame(conn, response)
            except (ConnectionError, json.JSONDecodeError, OSError):
                return

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class WorkerClient:
    def __init__(self, host: str, port: int, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

    def round_trip(self, payload: dict) -> dict:
        """One real socket round trip with an enforced timeout."""

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout_ms / 1000.0)
            sock.connect((self.host, self.port))
            _send_frame(sock, payload)
            return _recv_frame(sock)
        except (socket.timeout, TimeoutError) as exc:
            raise TimeoutError_(
                "worker round trip exceeded deadline",
                host=self.host,
                port=self.port,
                timeout_ms=self.timeout_ms,
            ) from exc
        finally:
            sock.close()

    def round_trip_with_retry(self, payload: dict, retries: int) -> tuple[dict, int]:
        """Retry idempotent attempts; returns the response and attempt count."""

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self.round_trip(payload), attempt
            except (TimeoutError_, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt == retries:
                    break
        raise last_error

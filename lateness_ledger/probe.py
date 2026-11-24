"""Loopback-only source subprocess sampling with genuine socket timeouts."""

from __future__ import annotations

import selectors
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import ProbeError


_SERVER_SCRIPT = r"""
import socket
import sys
import threading
import time

delay_ms = int(sys.argv[1])
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))
listener.listen(16)
print(f"PORT={listener.getsockname()[1]}", flush=True)

def serve(connection):
    try:
        connection.settimeout(5.0)
        connection.recv(64)
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        connection.sendall(b"OK")
    finally:
        connection.close()

try:
    while True:
        connection, _ = listener.accept()
        thread = threading.Thread(target=serve, args=(connection,), daemon=True)
        thread.start()
except OSError:
    pass
finally:
    listener.close()
"""


@dataclass
class ProbeAttempt:
    attempt: int
    elapsed_ms: float
    timed_out: bool
    error: str | None = None
    response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "timed_out": self.timed_out,
            "error": self.error,
            "response": self.response,
        }


@dataclass
class LoopbackProbeResult:
    ok: bool
    attempts: list[ProbeAttempt] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    retries: int = 0
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
            "retries": self.retries,
            "timed_out": self.timed_out,
        }


class LoopbackSourceSubprocess:
    """A local source process exposing a TCP sampling endpoint on loopback."""

    def __init__(self, *, server_delay_ms: int = 0) -> None:
        self.server_delay_ms = int(server_delay_ms)
        self.process: subprocess.Popen[bytes] | None = None
        self.host = "127.0.0.1"
        self.port = 0

    def start(self) -> "LoopbackSourceSubprocess":
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                _SERVER_SCRIPT,
                str(self.server_delay_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdout is not None
        if self.process.poll() is not None:
            stderr = (
                self.process.stderr.read().decode("utf-8", "replace")
                if self.process.stderr is not None
                else ""
            )
            raise ProbeError(
                "loopback source subprocess exited before publishing a port",
                details={"stderr": stderr},
            )
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        events = selector.select(timeout=5.0)
        selector.close()
        if not events:
            self.stop()
            raise ProbeError(
                "loopback source subprocess did not publish a port in time",
                details={"timeout_ms": 5000},
            )
        line = self.process.stdout.readline().decode("utf-8", "replace").strip()
        if not line.startswith("PORT="):
            self.stop()
            raise ProbeError(
                "loopback source subprocess returned an invalid port line",
                details={"line": line},
            )
        self.port = int(line.split("=", 1)[1])
        return self

    def probe(self, *, timeout_ms: int, attempts: int, backoff_ms: int) -> LoopbackProbeResult:
        if self.process is None or not self.port:
            raise ProbeError("loopback source subprocess has not been started")
        if timeout_ms <= 0 or attempts <= 0:
            raise ProbeError(
                "probe timeout and attempts must be positive",
                details={"timeout_ms": timeout_ms, "attempts": attempts},
            )
        outcomes: list[ProbeAttempt] = []
        overall_start = time.monotonic_ns()
        for attempt_number in range(1, attempts + 1):
            start = time.monotonic_ns()
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=timeout_ms / 1000.0
                ) as connection:
                    connection.settimeout(timeout_ms / 1000.0)
                    connection.sendall(b"PING\n")
                    payload = connection.recv(16)
                elapsed = (time.monotonic_ns() - start) / 1_000_000.0
                if not payload:
                    outcomes.append(
                        ProbeAttempt(
                            attempt=attempt_number,
                            elapsed_ms=elapsed,
                            timed_out=False,
                            error="empty response",
                        )
                    )
                else:
                    outcomes.append(
                        ProbeAttempt(
                            attempt=attempt_number,
                            elapsed_ms=elapsed,
                            timed_out=False,
                            response=payload.decode("utf-8", "replace"),
                        )
                    )
                    total = (time.monotonic_ns() - overall_start) / 1_000_000.0
                    return LoopbackProbeResult(
                        ok=True,
                        attempts=outcomes,
                        total_elapsed_ms=total,
                        retries=attempt_number - 1,
                        timed_out=False,
                    )
            except socket.timeout as exc:
                elapsed = (time.monotonic_ns() - start) / 1_000_000.0
                outcomes.append(
                    ProbeAttempt(
                        attempt=attempt_number,
                        elapsed_ms=elapsed,
                        timed_out=True,
                        error=str(exc),
                    )
                )
            except OSError as exc:
                elapsed = (time.monotonic_ns() - start) / 1_000_000.0
                outcomes.append(
                    ProbeAttempt(
                        attempt=attempt_number,
                        elapsed_ms=elapsed,
                        timed_out=False,
                        error=str(exc),
                    )
                )
            if attempt_number < attempts and backoff_ms > 0:
                time.sleep(backoff_ms / 1000.0)
        total = (time.monotonic_ns() - overall_start) / 1_000_000.0
        return LoopbackProbeResult(
            ok=False,
            attempts=outcomes,
            total_elapsed_ms=total,
            retries=max(0, attempts - 1),
            timed_out=any(item.timed_out for item in outcomes),
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()
        self.process = None

    def __enter__(self) -> "LoopbackSourceSubprocess":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.stop()

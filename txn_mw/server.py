"""Loopback TCP JSON-lines server."""

from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_json
from .engine import Engine
from .errors import MWError
from .logging import StructuredLogger


class TXNRequestHandler(socketserver.StreamRequestHandler):
    server: "TXNTCPServer"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(None)

    def _write(self, response: dict[str, Any]) -> None:
        data = (canonical_json(response) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def handle(self) -> None:
        while True:
            try:
                line = self.rfile.readline(self.server.engine.config.max_line_bytes + 1)
            except (ConnectionError, OSError):
                return
            if not line:
                return
            if len(line) > self.server.engine.config.max_line_bytes:
                self._safe_error(MWError("E_TXN", "JSON line exceeds maximum size", "$.line"))
                return
            if line in (b"\n", b"\r\n"):
                continue
            started_ns = time.monotonic_ns()
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise MWError("E_TXN", "request must be a JSON object", "$")
                self._dispatch(request, started_ns)
            except MWError as exc:
                self._safe_error(exc)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._safe_error(MWError("E_TXN", f"invalid JSON: {exc}", "$.json"))
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self.server.engine.logger.error(
                    "E_INTERNAL", "unhandled request failure", error=repr(exc)
                )
                self._safe_error(MWError("E_INTERNAL", "internal server error", "$"))

    def _dispatch(self, request: dict[str, Any], started_ns: int) -> None:
        op = request.get("op")
        if not isinstance(op, str):
            raise MWError("E_TXN", "missing operation", "$.op")
        engine = self.server.engine
        if op == "ping":
            self._write({"ok": True, "op": "ping", "service": "txn-mw", "format_version": FORMAT_VERSION})
            return
        if op == "half":
            response = engine.persist_half(request)
            # Persistence must be complete before writing the acknowledgment.
            self._write(response)
            # The durable query-eligibility marker deliberately follows the
            # socket send. A crash in between leaves the half orphaned rather
            # than illegally querying an unacknowledged message.
            engine.mark_half_ack(response["txn_id"])
            return
        if op == "commit":
            self._write(engine.producer_command(request, "commit"))
            return
        if op == "rollback":
            self._write(engine.producer_command(request, "rollback"))
            return
        if op == "pull":
            self._write(engine.pull(request, started_ns=started_ns))
            return
        if op == "trace":
            txn_id = request.get("txn_id")
            if not isinstance(txn_id, str):
                raise MWError("E_TXN", "txn_id is required", "$.txn_id")
            self._write(engine.transaction_trace(txn_id))
            return
        if op in {"status", "inventory"}:
            self._write(engine.inventory())
            return
        if op in {"latency", "stats"}:
            txn_id = request.get("txn_id")
            operation = request.get("operation")
            if txn_id is not None and not isinstance(txn_id, str):
                raise MWError("E_TXN", "txn_id must be a string", "$.txn_id")
            self._write(engine.latency_stats(txn_id=txn_id, operation=operation))
            return
        raise MWError("E_TXN", "unknown operation", "$.op")

    def _safe_error(self, exc: MWError) -> None:
        self.server.engine.logger.error(
            exc.code,
            exc.message,
            path=exc.path,
            details=exc.details,
            remote=self.client_address,
        )
        try:
            self._write(exc.as_dict())
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class TXNTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: Engine, *, logger: StructuredLogger | None = None):
        self.engine = engine
        self.structured_logger = logger or engine.logger
        super().__init__(address, TXNRequestHandler)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server_address[:2]
        return str(host), int(port)

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="txn-tcp-server", daemon=True)
        thread.start()
        return thread


class RunningServer:
    """Convenience owner for engine + TCP server lifecycle."""

    def __init__(self, engine: Engine, host: str = "127.0.0.1", port: int = 0):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("the TCP server is restricted to loopback")
        self.engine = engine
        self.server = TXNTCPServer((host, port), engine, logger=engine.logger)
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self.server.address

    def start(self) -> None:
        self.thread = self.server.start_background()
        self.engine.start()

    def stop(self, *, report: bool = True) -> dict[str, Any] | None:
        self.server.shutdown()
        self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)
        return self.engine.shutdown(report=report)

    def __enter__(self) -> "RunningServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop(report=True)


def request(host: str, port: int, payload: dict[str, Any], *, timeout: float = 3.0) -> dict[str, Any]:
    """Small synchronous helper shared by the CLI, tools and tests."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((canonical_json(payload) + "\n").encode("utf-8"))
        buffer = bytearray()
        while b"\n" not in buffer:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buffer.extend(chunk)
    line = bytes(buffer).split(b"\n", 1)[0]
    if not line:
        raise ConnectionError("server returned no JSON line")
    return json.loads(line.decode("utf-8"))


from __future__ import annotations

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _DeterministicHandler(BaseHTTPRequestHandler):
    server: "DeterministicHTTPServer"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._respond()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self._respond()

    def _respond(self) -> None:
        if self.server.mode == "fixed_delay":
            time.sleep(self.server.processing_ms / 1000.0)
        lease_id = self.headers.get("X-Lease-Id", "")
        request_seq = self.headers.get("X-Request-Seq", "0")
        status = 200 if lease_id else 400
        body = {
            "source_id": self.server.source_id,
            "mode": self.server.mode,
            "lease_id": lease_id,
            "request_seq": int(request_seq) if str(request_seq).isdigit() else 0,
            "processing_ms": self.server.processing_ms,
            "status": "success" if status == 200 else "missing_lease",
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


class DeterministicHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        source_id: str,
        mode: str,
        processing_ms: int,
    ) -> None:
        self.source_id = source_id
        self.mode = mode
        self.processing_ms = processing_ms
        super().__init__(address, _DeterministicHandler)


def serve_source(source_id: str, port: int, mode: str, processing_ms: int = 0) -> None:
    if mode == "refuse":
        # No listener is opened. Clients receive a genuine connection refusal on loopback.
        while True:
            time.sleep(1.0)
    if mode not in {"success", "fixed_delay"}:
        raise ValueError(f"unsupported mode: {mode}")
    server = DeterministicHTTPServer(("127.0.0.1", port), source_id, mode, processing_ms)
    stop_event = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, _stop)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, _stop)
    except ValueError:
        # Signal registration is unavailable outside the main thread.
        pass
    thread = threading.Thread(target=server.serve_forever, name=f"mock-{source_id}", daemon=True)
    thread.start()
    try:
        while not stop_event.wait(0.25):
            pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

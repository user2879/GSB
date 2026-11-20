"""Shared threaded HTTP/1.1 loopback target used by both local services."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def make_handler(role: str, default_delay_ms: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parts = urlsplit(self.path)
            delay_ms = default_delay_ms
            status = 200
            if "/__slow__" in parts.path:
                delay_ms = max(delay_ms, 300)
            elif role == "mirror" and "/__mirror_slow__" in parts.path:
                delay_ms = max(delay_ms, 1200)
            elif "/__error__" in parts.path:
                status = 503
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            body = json.dumps(
                {
                    "role": role,
                    "path": parts.path,
                    "request_id": self.headers.get("X-Request-Id", ""),
                    "pool_id": self.headers.get("X-Pool-Id", ""),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Target-Role", role)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, fmt: str, *args: object) -> None:
            # Keep deterministic local runs quiet; the arbiter ledger is the
            # source of truth for observed requests.
            return

    return Handler


def make_server(host: str, port: int, role: str, delay_ms: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(role, delay_ms))
    server.daemon_threads = True
    return server


def serve_forever(servers: list[ThreadingHTTPServer]) -> None:
    threads = []
    for server in servers:
        thread = threading.Thread(target=server.serve_forever, daemon=False)
        thread.start()
        threads.append(thread)
    try:
        for thread in threads:
            thread.join()
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def parse_common(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--delay-ms", type=int, default=15)
    return parser.parse_args()

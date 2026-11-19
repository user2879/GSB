from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


STATE_LOCK = threading.Lock()
REQUEST_COUNTS: dict[tuple[int, str], int] = {}
ATTEMPT_VALUES: dict[tuple[int, str], list[int]] = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MeshDiagMock/1"

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        port = int(self.server.server_address[1])
        path = self.path.split("?", 1)[0]
        if path == "/__admin/reset":
            with STATE_LOCK:
                REQUEST_COUNTS.clear()
                ATTEMPT_VALUES.clear()
            self._send(200, b"reset\n")
            return
        if path == "/__admin/state":
            with STATE_LOCK:
                rows = [
                    {
                        "port": key[0],
                        "path": key[1],
                        "count": count,
                        "attempts": list(ATTEMPT_VALUES.get(key, [])),
                    }
                    for key, count in sorted(REQUEST_COUNTS.items())
                ]
            body = (json.dumps(rows, sort_keys=True) + "\n").encode("utf-8")
            self._send(200, body)
            return
        attempt_header = self.headers.get("X-Mesh-Attempt", "1")
        try:
            attempt_no = max(1, int(attempt_header))
        except ValueError:
            attempt_no = 1
        key = (port, path)
        with STATE_LOCK:
            REQUEST_COUNTS[key] = REQUEST_COUNTS.get(key, 0) + 1
            ATTEMPT_VALUES.setdefault(key, []).append(attempt_no)
        if "/slow" in path:
            time.sleep(0.22)
            self._send(200, b"slow-but-successful\n")
            return
        if "/fail" in path:
            self._send(503, b"injected-failure\n")
            return
        if "/flaky" in path:
            if attempt_no == 1:
                self._send(503, b"first-attempt-failure\n")
            else:
                self._send(200, b"retry-success\n")
            return
        if "/timeout_once" in path:
            if attempt_no == 1:
                time.sleep(0.22)
                self._send(200, b"late-success\n")
            else:
                self._send(200, b"retry-success\n")
            return
        self._send(200, b"ok\n")


def start_servers(ports: list[int]) -> list[ThreadingHTTPServer]:
    servers = []
    for port in ports:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        servers.append(server)
    return servers


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback upstream")
    parser.add_argument(
        "--ports",
        default="18181,18182,18183,18184,18185,18186,18187,18188",
        help="comma-separated loopback ports",
    )
    args = parser.parse_args()
    ports = sorted({int(value.strip()) for value in args.ports.split(",") if value.strip()})
    servers = start_servers(ports)
    threads = []
    for server in servers:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
    for port in ports:
        print(f"READY 127.0.0.1:{port}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

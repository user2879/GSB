"""Loopback TCP decode executor hosted in a child process."""

from __future__ import annotations

import argparse
import base64
import json
import socketserver
import time
from typing import Any

from .codec import decode_block_bytes


def decode_request(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    delay_ms = int(request.get("response_delay_ms", request.get("delay_ms", 0)) or 0)
    if delay_ms:
        time.sleep(delay_ms / 1000.0)

    block_group_id = request["block_group_id"]
    decoded_rows = int(request.get("row_count", 0) or 0)
    if request.get("block_b64"):
        block = decode_block_bytes(base64.b64decode(request["block_b64"]))
        decoded_rows = block["row_count"]
    elif request.get("block_bytes_hex"):
        block = decode_block_bytes(bytes.fromhex(request["block_bytes_hex"]))
        decoded_rows = block["row_count"]
    elif request.get("rows") is not None:
        decoded_rows = len(request["rows"])

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return {
        "block_group_id": block_group_id,
        "decoded_rows": decoded_rows,
        "elapsed_ms": elapsed_ms,
        "transport": "tcp/127.0.0.1",
    }


class ThreadingTCPRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            try:
                request = json.loads(line.decode("utf-8"))
                if request.get("type") == "ping":
                    response = {"ok": True, "pong": True}
                elif request.get("type") == "shutdown":
                    response = {"ok": True, "shutting_down": True}
                    self.wfile.write((json.dumps(response) + "\n").encode())
                    self.wfile.flush()
                    self.server.shutdown()
                    return
                else:
                    receipt = decode_request(request)
                    response = {"ok": True, "receipt": receipt}
            except Exception as exc:  # noqa: BLE001 - boundary must report every error
                response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
            self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode())
            self.wfile.flush()


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str = "127.0.0.1", port: int = 0) -> dict[str, Any]:
    server = ReusableThreadingTCPServer((host, port), ThreadingTCPRequestHandler)
    actual_host, actual_port = server.server_address[:2]
    ready = {"ready": True, "host": actual_host, "port": actual_port, "transport": "tcp/127.0.0.1"}
    print(json.dumps(ready, sort_keys=True), flush=True)
    try:
        server.serve_forever(poll_interval=0.02)
    finally:
        server.server_close()
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()

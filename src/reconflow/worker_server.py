from __future__ import annotations

import argparse
import json
import socketserver
import time
from typing import Any

from .canonical import digest_value
from .pipeline import compute_domain_dedup


class ShardTCPHandler(socketserver.BaseRequestHandler):
    def _readline(self) -> bytes:
        return self.rfile.readline()

    def handle(self) -> None:
        server: "ShardTCPServer" = self.server  # type: ignore[assignment]
        self.request.settimeout(server.socket_timeout_seconds)
        self.rfile = self.request.makefile("rwb")
        self.wfile = self.request.makefile("rwb")

        hello = {"type": "hello", "protocol": 1, "server": "reconflow-shard-worker", "pid": __import__("os").getpid()}
        self.wfile.write((json.dumps(hello, sort_keys=True) + "\n").encode())
        self.wfile.flush()

        try:
            line = self._readline()
            if not line:
                return
            handshake = json.loads(line)
            if handshake.get("type") != "handshake" or handshake.get("protocol") != 1:
                self._send({"type": "error", "code": "E_FIELD", "message": "invalid handshake"})
                return
            self._send({"type": "handshake_ack", "ok": True, "protocol": 1})

            while True:
                line = self._readline()
                if not line:
                    break
                task = json.loads(line)
                if task.get("type") == "shutdown":
                    self._send({"type": "shutdown_ack", "ok": True})
                    break
                if server.delay_seconds:
                    time.sleep(server.delay_seconds)
                result = compute_domain_dedup(
                    str(task["domain_id"]),
                    list(task["docs"]),
                    int(task["similarity_threshold_bp"]),
                    dict(task["stage_params"]),
                )
                receipt: dict[str, Any] = {
                    "type": "receipt",
                    "task_id": task["task_id"],
                    "ok": True,
                    "result": result,
                    "digest": digest_value(result),
                }
                if server.inject_missing_field and server.inject_missing_field not in ("", "none"):
                    field = server.inject_missing_field
                    if field in receipt:
                        del receipt[field]
                    elif field in result:
                        del receipt["result"][field]
                self._send(receipt)
        except (TimeoutError, OSError):
            return
        except Exception as exc:  # pragma: no cover - defensive wire boundary
            try:
                self._send({"type": "error", "code": getattr(exc, "code", "E_WORKER"), "message": str(exc)})
            except OSError:
                pass

    def _send(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload, sort_keys=True) + "\n").encode())
        self.wfile.flush()


class ShardTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[ShardTCPHandler], *, timeout_ms: int, delay_ms: int, inject_missing_field: str) -> None:
        super().__init__(address, handler)
        self.socket_timeout_seconds = max(0.001, timeout_ms / 1000.0)
        self.delay_seconds = max(0.0, delay_ms / 1000.0)
        self.inject_missing_field = inject_missing_field


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ReconFlow shard compute TCP worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--inject-missing-field", default="")
    args = parser.parse_args(argv)

    try:
        server = ShardTCPServer(
            (args.host, args.port),
            ShardTCPHandler,
            timeout_ms=args.timeout_ms,
            delay_ms=args.delay_ms,
            inject_missing_field=args.inject_missing_field,
        )
    except OSError as exc:
        print(json.dumps({"event": "error", "code": "E_PORT", "message": str(exc)}), flush=True)
        return 2

    host, port = server.server_address[:2]
    print(json.dumps({"event": "ready", "host": host, "port": port, "protocol": 1}), flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

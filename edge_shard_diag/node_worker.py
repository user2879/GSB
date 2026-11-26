from __future__ import annotations

import argparse
import socketserver
import threading
import time
from typing import Any

from .aggregate import aggregate_shard
from .errors import Codes
from .tcpio import read_line, send_line


class NodeHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "NodeServer" = self.server  # type: ignore[assignment]
        self.connection.settimeout(server.socket_timeout)
        while True:
            try:
                request = read_line(self.connection, timeout=server.socket_timeout)
            except Exception:
                return
            request_type = request.get("type")
            if request_type == "handshake":
                send_line(
                    self.connection,
                    {
                        "ok": True,
                        "type": "hello",
                        "node_id": server.node_id,
                        "pid": __import__("os").getpid(),
                        "monotonic_ms": time.monotonic_ns() // 1_000_000,
                    },
                )
                continue
            if request_type == "shutdown":
                send_line(self.connection, {"ok": True, "type": "shutdown_ack"})
                threading.Timer(0.05, server.shutdown).start()
                return
            if request_type == "renew":
                send_line(self.connection, server.renew(request))
                continue
            if request_type in {"lease_grant", "grant_lease"}:
                lease_id = str(request.get("lease_id", ""))
                version = int(request.get("lease_version", 1))
                with server.lock:
                    server.lease_versions[lease_id] = version
                send_line(self.connection, {"ok": True, "lease_id": lease_id, "lease_version": version, "type": "lease_grant_ack"})
                continue
            if request_type in {"shard", "dispatch"}:
                send_line(self.connection, server.dispatch(request))
                continue
            send_line(self.connection, {"ok": False, "error": {"code": Codes.FORMAT, "message": f"unknown request type: {request_type}"}})


class NodeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler: type[NodeHandler], *, node_id: str, fixed_scale: int) -> None:
        super().__init__(server_address, handler)
        self.node_id = node_id
        self.fixed_scale = fixed_scale
        self.socket_timeout = 5.0
        self.lease_versions: dict[str, int] = {}
        self.lock = threading.RLock()

    def renew(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(request.get("lease_id", ""))
        expected = request.get("lease_version")
        with self.lock:
            current = self.lease_versions.get(lease_id)
            if current is None or not isinstance(expected, int) or expected != current:
                return {
                    "ok": False,
                    "error": {
                        "code": Codes.LEASE,
                        "message": "optimistic lease version mismatch",
                        "field": "lease_version",
                        "current_version": current,
                        "expected_version": expected,
                    },
                }
            new_version = current + 1
            self.lease_versions[lease_id] = new_version
            return {"ok": True, "lease_id": lease_id, "lease_version": new_version, "renew_count": new_version}

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        start_ns = time.monotonic_ns()
        lease_id = str(request.get("lease_id", ""))
        lease_version = int(request.get("lease_version", 0))
        with self.lock:
            self.lease_versions[lease_id] = lease_version
        shard = request.get("shard") or {"id": request.get("shard_id"), "sample_ids": [s["id"] for s in request.get("samples", [])]}
        samples = {sample["id"]: sample for sample in request.get("samples", [])}
        result = aggregate_shard(
            shard,
            samples,
            capacity_version=int(request.get("capacity_version", 0)),
            scale_factor=int(request.get("scale_factor", 1)),
            node_fixed_scale=self.fixed_scale,
        )
        elapsed_ns = time.monotonic_ns() - start_ns
        result.update(
            {
                "ok": True,
                "type": "shard_result",
                "node_id": self.node_id,
                "lease_id": lease_id,
                "lease_version": lease_version,
                "duration_ms": elapsed_ns / 1_000_000.0,
                "duration_monotonic_start_ms": start_ns // 1_000_000,
                "duration_monotonic_end_ms": time.monotonic_ns() // 1_000_000,
            }
        )
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edge shard JSON-lines node subprocess")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--fixed-scale", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        server = NodeServer((args.host, args.port), NodeHandler, node_id=args.node_id, fixed_scale=args.fixed_scale)
    except OSError as exc:
        print(f"[{Codes.PORT}] node={args.node_id} bind={args.host}:{args.port} {exc}", flush=True)
        return 2
    print(f"NODE_READY {args.node_id} {args.host}:{args.port} pid={__import__('os').getpid()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

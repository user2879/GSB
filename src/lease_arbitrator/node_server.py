from __future__ import annotations

import argparse
import json
import random
import socketserver
from dataclasses import dataclass, field
from typing import Any

from .model import monotonic_ms
from .protocol import PROTOCOL_VERSION, dumps_line


@dataclass
class NodeLease:
    lease_id: str
    units: int
    generation: int = 0
    state: str = "new"
    history: list[dict[str, Any]] = field(default_factory=list)


class NodeCore:
    def __init__(self, node_id: str, seed: int):
        self.node_id = node_id
        self.seed = seed
        self.rng = random.Random(seed)
        self.instance_nonce = self.rng.getrandbits(32)
        self.leases: dict[str, NodeLease] = {}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        now = monotonic_ms()
        if request_type == "ping":
            return {"ok": True, "type": "pong", "node_id": self.node_id, "monotonic_ms": now}
        if request_type == "hello":
            return {
                "ok": True,
                "type": "hello_ack",
                "protocol": PROTOCOL_VERSION,
                "node_id": self.node_id,
                "seed": self.seed,
                "instance_nonce": self.instance_nonce,
                "monotonic_ms": now,
            }
        if request_type == "apply":
            return self._apply(request, now)
        if request_type in {"renew", "release", "revoke"}:
            return self._cas(request, now)
        return {"ok": False, "error": "E_PROTOCOL", "message": f"unknown type {request_type!r}"}

    def _apply(self, request: dict[str, Any], now: int) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            return {"ok": False, "error": "E_SPEC", "message": "lease_id required"}
        units = request.get("units")
        if not isinstance(units, int) or isinstance(units, bool) or units <= 0:
            return {"ok": False, "error": "E_SPEC", "message": "units must be positive integer"}
        if lease_id in self.leases:
            existing = self.leases[lease_id]
            return {
                "ok": False,
                "error": "E_EXISTS",
                "message": "lease already exists on node",
                "lease_id": lease_id,
                "generation": existing.generation,
            }
        requested_generation = request.get("generation", 1)
        if requested_generation != 1:
            return {
                "ok": False,
                "error": "E_GEN",
                "message": "new lease must start at generation 1",
                "lease_id": lease_id,
                "received_generation": requested_generation,
            }
        lease = NodeLease(lease_id=lease_id, units=units, generation=1, state="granted")
        lease.history.append({"to_state": "granted", "generation_after": 1, "monotonic_ms": now})
        self.leases[lease_id] = lease
        return {
            "ok": True,
            "type": "apply_ack",
            "lease_id": lease_id,
            "node_id": self.node_id,
            "state": "granted",
            "generation_used": 1,
            "generation_before": 0,
            "generation": 1,
            "generation_after": 1,
            "monotonic_ms": now,
        }

    def _cas(self, request: dict[str, Any], now: int) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        action = request.get("type")
        if not isinstance(lease_id, str) or not lease_id:
            return {"ok": False, "error": "E_SPEC", "message": "lease_id required"}
        lease = self.leases.get(lease_id)
        if lease is None:
            return {
                "ok": False,
                "error": "E_NOLEASE",
                "message": "node does not recognize lease",
                "lease_id": lease_id,
            }
        received = request.get("generation")
        if not isinstance(received, int) or isinstance(received, bool):
            return {"ok": False, "error": "E_SPEC", "message": "generation must be integer"}
        before = lease.generation
        if received != before:
            return {
                "ok": False,
                "error": "E_STALE",
                "message": "generation label does not match current node label",
                "lease_id": lease_id,
                "action": action,
                "generation_used": received,
                "generation_before": before,
                "current_generation": before,
                "generation_after": before,
                "monotonic_ms": now,
            }
        after = before + 1
        lease.generation = after
        if action == "renew":
            lease.state = "renewed"
        elif action == "release":
            lease.state = "released"
        elif action == "revoke":
            lease.state = "revoked"
        lease.history.append(
            {
                "to_state": lease.state,
                "generation_before": before,
                "generation_after": after,
                "monotonic_ms": now,
            }
        )
        return {
            "ok": True,
            "type": f"{action}_ack",
            "lease_id": lease_id,
            "node_id": self.node_id,
            "state": lease.state,
            "generation_used": before,
            "generation_before": before,
            "generation": after,
            "generation_after": after,
            "monotonic_ms": now,
        }


class NodeRequestHandler(socketserver.StreamRequestHandler):
    timeout = 5.0

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.timeout)

    def handle(self) -> None:
        core: NodeCore = self.server.core  # type: ignore[attr-defined]
        while True:
            try:
                line = self.rfile.readline()
            except (TimeoutError, OSError):
                break
            if not line:
                break
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be object")
                response = core.handle(request)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                response = {"ok": False, "error": "E_PROTOCOL", "message": str(exc)}
            self.wfile.write(dumps_line(response))
            self.wfile.flush()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(node_id: str, host: str, port: int, seed: int, timeout: float = 5.0) -> None:
    if host != "127.0.0.1":
        raise ValueError("node server may bind only 127.0.0.1")
    server = ThreadedTCPServer((host, port), NodeRequestHandler)
    server.core = NodeCore(node_id, seed)  # type: ignore[attr-defined]
    NodeRequestHandler.timeout = timeout
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lease arbitration JSON-line node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    serve(args.node_id, args.host, args.port, args.seed, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

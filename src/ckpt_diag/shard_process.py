"""Training shard subprocess: loopback JSONL server.

Each shard is an independent OS process. It owns its lease authority and runs
the integer fixed-point compute from :mod:`ckpt_diag.compute`, reporting a
monotonic-clock millisecond duration for every batch.
"""
from __future__ import annotations

import argparse
import json
import socket
import socketserver
import sys
from pathlib import Path

from .clock import MonotonicClock
from .compute import compute_batch
from .errors import DiagError
from .lease import LeaseManager
from .protocol import HOST, recv_message, send_message
from .spec import ShardSpec, load_spec


class ShardState:
    def __init__(self, shard: ShardSpec, ttl_ms: int) -> None:
        self.shard = shard
        self.ttl_ms = ttl_ms
        self.clock = MonotonicClock()
        self.leases = LeaseManager(self.clock.now_ms)


def _handle(message: dict, state: ShardState) -> dict:
    action = message.get("action")
    shard = state.shard
    if action == "ping":
        return {"ok": True, "shard_id": shard.shard_id}
    if action == "issue_lease":
        lease = state.leases.issue(shard.shard_id, state.ttl_ms, message["batch_id"])
        return {"ok": True, "lease": lease.__dict__}
    if action == "renew_lease":
        try:
            lease = state.leases.renew(message["lease_id"],
                                       message["expected_version"], state.ttl_ms)
        except DiagError as exc:
            return {"ok": False, "error": exc.to_dict()}
        return {"ok": True, "lease": lease.__dict__}
    if action == "release_lease":
        lease = state.leases.release(message["lease_id"],
                                     message.get("expected_version"))
        return {"ok": True, "lease": lease.__dict__}
    if action == "run_batch":
        lease_id = message["lease_id"]
        if not state.leases.is_active(lease_id):
            return {"ok": False, "error": {
                "code": "E_LEASE", "message": "lease missing or expired at run time",
                "location": f"lease:{lease_id}", "field": "expires_ms", "detail": None}}
        start = state.clock.now_ms()
        result = compute_batch(
            shard, message["batch_id"], int(message["param_version"]),
            tuple(message.get("sample_ids", ())),
            tuple(message.get("scope_layers", ())))
        elapsed = state.clock.now_ms() - start
        return {"ok": True, "result": result.to_dict(), "elapsed_ms": elapsed,
                "lease_version": state.leases.get(lease_id).lease_version}
    return {"ok": False, "error": {"code": "E_FORMAT",
                                    "message": f"unknown action {action!r}",
                                    "location": "<request>", "field": "action",
                                    "detail": None}}


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: ShardState = self.server.state  # type: ignore[attr-defined]
        raw = self.rfile.readline()
        if not raw:
            return
        message = json.loads(raw.decode("utf-8"))
        try:
            response = _handle(message, state)
        except DiagError as exc:
            response = {"ok": False, "error": exc.to_dict()}
        send_message(self.wfile, response)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, state: ShardState) -> None:
        super().__init__(address, handler)
        self.state = state


def serve(shard: ShardSpec, ttl_ms: int) -> _Server:
    server = _Server((shard.host, shard.port), _Handler, ShardState(shard, ttl_ms))
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run one training shard subprocess")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--ttl-ms", type=int, default=5000)
    args = parser.parse_args(argv)
    spec = load_spec(Path(args.spec))
    shard = spec.shard(args.shard_id)
    server = serve(shard, args.ttl_ms)
    print(json.dumps({"ready": True, "shard_id": shard.shard_id,
                      "host": shard.host, "port": shard.port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

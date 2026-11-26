"""Independent node subprocess listening on loopback TCP/JSON-lines."""

from __future__ import annotations

import argparse
import json
import socketserver
import time
from typing import Any

from .samples import aggregate_batch


class NodeState:
    def __init__(self, node_id: str, seed: int, scale: int, capacity_version: int) -> None:
        self.node_id = node_id
        self.seed = seed
        self.scale = scale
        self.capacity_version = capacity_version
        self.leases: dict[str, dict[str, Any]] = {}

    def hello(self) -> dict[str, Any]:
        return {
            "ok": True,
            "type": "hello_ack",
            "node_id": self.node_id,
            "protocol_version": 1,
            "seed": self.seed,
            "scale": self.scale,
            "capacity_version": self.capacity_version,
        }

    def open_lease(self, message: dict[str, Any]) -> dict[str, Any]:
        required = ("lease_id", "shard_id", "ttl_ms")
        missing = [key for key in required if key not in message]
        if missing:
            return error_response("E_FIELD", f"open_lease missing {missing}", missing=missing)
        lease_id = str(message["lease_id"])
        now_ns = time.monotonic_ns()
        ttl_ns = max(1, int(message["ttl_ms"])) * 1_000_000
        lease = {
            "lease_id": lease_id,
            "shard_id": str(message["shard_id"]),
            "version": 1,
            "renew_count": 0,
            "capacity_version": int(message.get("capacity_version", self.capacity_version)),
            "expires_ns": now_ns + ttl_ns,
        }
        self.leases[lease_id] = lease
        return {"ok": True, **{key: lease[key] for key in ("lease_id", "shard_id", "version", "renew_count", "capacity_version")}}

    def renew_lease(self, message: dict[str, Any]) -> dict[str, Any]:
        lease_id = message.get("lease_id")
        expected_version = message.get("version")
        max_renewals = int(message.get("max_renewals", 0))
        ttl_ms = max(1, int(message.get("ttl_ms", 1000)))
        lease = self.leases.get(str(lease_id))
        if lease is None:
            return error_response("E_LEASE", f"unknown lease {lease_id!r}", field_name="lease_id")
        if not isinstance(expected_version, int) or isinstance(expected_version, bool):
            return error_response("E_FIELD", "version must be an integer", field_name="version")
        if lease["version"] != expected_version:
            return error_response(
                "E_LEASE",
                f"lease version mismatch: expected {lease['version']}, got {expected_version}",
                field_name="version",
                details={"current_version": lease["version"], "provided_version": expected_version},
            )
        if time.monotonic_ns() > int(lease["expires_ns"]):
            return error_response("E_LEASE", "lease has expired", field_name="expires_at_ms")
        if int(lease["renew_count"]) >= max_renewals:
            return error_response(
                "E_RENEW",
                f"renewal limit {max_renewals} reached",
                field_name="renew_count",
                details={"renew_count": lease["renew_count"], "max_renewals": max_renewals},
            )
        lease["version"] += 1
        lease["renew_count"] += 1
        lease["expires_ns"] = time.monotonic_ns() + ttl_ms * 1_000_000
        return {
            "ok": True,
            "lease_id": lease_id,
            "version": lease["version"],
            "renew_count": lease["renew_count"],
            "expires_in_ms": ttl_ms,
        }

    def release_lease(self, message: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(message.get("lease_id", ""))
        self.leases.pop(lease_id, None)
        return {"ok": True, "lease_id": lease_id, "released": True}

    def aggregate(self, message: dict[str, Any]) -> dict[str, Any]:
        for field_name in ("shard_id", "capacity_version", "batch"):
            if field_name not in message:
                return error_response("E_FIELD", f"aggregate missing {field_name}", field_name=field_name)
        shard_id = message["shard_id"]
        capacity_version = int(message["capacity_version"])
        if capacity_version != self.capacity_version:
            return error_response(
                "E_VERSION",
                f"node capacity version is {self.capacity_version}, request used {capacity_version}",
                field_name="capacity_version",
                details={"current": self.capacity_version, "requested": capacity_version},
            )
        delay_ms = int(message.get("delay_ms", 0) or 0)
        if delay_ms > 0:
            time.sleep(min(delay_ms, 30_000) / 1000.0)
        start_ns = time.monotonic_ns()
        try:
            summary = aggregate_batch(message["batch"], self.scale)
        except Exception as exc:  # node reports malformed batch as field error
            return error_response("E_FIELD", f"cannot aggregate batch: {exc}", field_name="batch")
        elapsed_ms = max(0, (time.monotonic_ns() - start_ns) // 1_000_000)
        return {
            "ok": True,
            "node_id": self.node_id,
            "shard_id": shard_id,
            "capacity_version": capacity_version,
            "summary": summary,
            "elapsed_ms": int(elapsed_ms),
            "monotonic_clock": "time.monotonic_ns",
        }


def error_response(
    code: str,
    message: str,
    *,
    field_name: str | None = None,
    missing: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "code": code, "message": message, "field": field_name}
    if missing is not None:
        payload["missing"] = missing
    if details is not None:
        payload["details"] = details
    return payload


def make_handler(state: NodeState, timeout_ms: int):
    class NodeHandler(socketserver.StreamRequestHandler):
        timeout = max(0.1, timeout_ms / 1000.0)

        def handle(self) -> None:
            self.connection.settimeout(self.timeout)
            line = self.rfile.readline(64 * 1024 * 1024)
            if not line:
                return
            try:
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError("message is not an object")
            except Exception as exc:
                response = error_response("E_FIELD", f"invalid JSON line: {exc}", field_name="json")
            else:
                response = self.dispatch(message)
            self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
            self.wfile.flush()

        def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
            message_type = message.get("type")
            if message_type == "hello":
                return state.hello()
            if message_type == "lease_open":
                return state.open_lease(message)
            if message_type == "lease_renew":
                return state.renew_lease(message)
            if message_type == "lease_release":
                return state.release_lease(message)
            if message_type == "aggregate":
                return state.aggregate(message)
            return error_response("E_FIELD", f"unknown message type {message_type!r}", field_name="type")

    return NodeHandler


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ShardMesh node subprocess")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--capacity-version", type=int, required=True)
    parser.add_argument("--request-timeout-ms", type=int, default=5000)
    args = parser.parse_args(argv)

    state = NodeState(args.node_id, args.seed, args.scale, args.capacity_version)
    try:
        server = ThreadedTCPServer((args.host, args.port), make_handler(state, args.request_timeout_ms))
    except OSError as exc:
        print(
            json.dumps(
                error_response(
                    "E_PORT",
                    f"node {args.node_id} cannot bind {args.host}:{args.port}: {exc}",
                    field_name="port",
                    details={"errno": exc.errno},
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    with server:
        print(
            json.dumps(
                {
                    "type": "ready",
                    "node_id": args.node_id,
                    "host": args.host,
                    "port": args.port,
                    "capacity_version": args.capacity_version,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.05)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

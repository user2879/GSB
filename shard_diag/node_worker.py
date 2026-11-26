from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading
import time
from typing import Any

from . import AGGREGATION_CALIBER_VERSION, PROTOCOL_VERSION
from .determinism import aggregate_batch


class NodeState:
    def __init__(self, node_id: str, seed: int, quota_version: int, scale: int) -> None:
        self.node_id = node_id
        self.seed = seed
        self.quota_version = quota_version
        self.scale = scale
        self.lock = threading.RLock()
        self.leases: dict[str, dict[str, Any]] = {}

    def grant(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = required(payload, "lease_id")
        shard_id = required(payload, "shard_id")
        version = int(payload.get("lease_version", 1))
        with self.lock:
            existing = self.leases.get(lease_id)
            if existing and existing.get("state") == "active":
                return error("E_LEASE", f"lease {lease_id} already exists", "lease_id", lease_id)
            self.leases[lease_id] = {
                "lease_id": lease_id,
                "shard_id": shard_id,
                "version": version,
                "state": "active",
                "expires_monotonic_ms": payload.get("expires_monotonic_ms"),
            }
        return {"ok": True, "lease_id": lease_id, "lease_version": version}

    def renew(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = required(payload, "lease_id")
        expected = int(required(payload, "expected_version"))
        proposed = int(required(payload, "proposed_version"))
        expires = required(payload, "expires_monotonic_ms")
        with self.lock:
            lease = self.leases.get(lease_id)
            current = None if lease is None else int(lease["version"])
            state = None if lease is None else lease.get("state")
            if lease is None or state != "active" or current != expected:
                return error(
                    "E_LEASE",
                    "optimistic lease version check failed; expiry was not extended",
                    "lease_version",
                    lease_id,
                    {"expected_version": expected, "current_version": current, "state": state},
                )
            lease["version"] = proposed
            lease["expires_monotonic_ms"] = expires
            return {
                "ok": True,
                "lease_id": lease_id,
                "lease_version": proposed,
                "expires_monotonic_ms": expires,
                "renew_count": lease.get("renew_count", 0) + 1,
            }

    def release(self, payload: dict[str, Any], *, revoked: bool = False) -> dict[str, Any]:
        lease_id = required(payload, "lease_id")
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is not None:
                lease["state"] = "revoked_rollback" if revoked else "released"
        return {"ok": True, "lease_id": lease_id, "state": "revoked_rollback" if revoked else "released"}


def required(payload: dict[str, Any], field: str) -> Any:
    if field not in payload:
        raise MissingField(field)
    return payload[field]


class MissingField(Exception):
    def __init__(self, field: str) -> None:
        super().__init__(f"missing field {field}")
        self.field = field


def error(code: str, message: str, field: str | None = None, object_id: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "code": code, "message": message}
    if field:
        payload["field"] = field
    if object_id:
        payload["object_id"] = object_id
    if details:
        payload["details"] = details
    return payload


class NodeHandler(socketserver.StreamRequestHandler):
    timeout = 5.0

    def handle(self) -> None:
        self.connection.settimeout(self.timeout)
        state: NodeState = self.server.state  # type: ignore[attr-defined]
        try:
            hello_line = self.rfile.readline()
            if not hello_line:
                return
            hello = json.loads(hello_line.decode("utf-8"))
            if hello.get("type") != "hello":
                self.write({"ok": False, "code": "E_FIELD", "field": "type", "message": "expected hello"})
                return
            self.write({
                "ok": True,
                "type": "hello_ack",
                "node_id": state.node_id,
                "protocol_version": PROTOCOL_VERSION,
                "caliber_version": AGGREGATION_CALIBER_VERSION,
            })
            line = self.rfile.readline()
            if not line:
                return
            request = json.loads(line.decode("utf-8"))
            self.write(dispatch(state, request, self.server))
        except MissingField as exc:
            self.write(error("E_FIELD", str(exc), field=exc.field, object_id=state.node_id))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.write(error("E_FIELD", f"invalid JSON line: {exc}", field="request"))
        except (socket.timeout, TimeoutError) as exc:
            self.write(error("E_TIMEOUT", f"node socket timeout: {exc}", field="socket.read"))
        except Exception as exc:  # pragma: no cover - defensive subprocess boundary
            self.write(error("E_INTERNAL", f"{type(exc).__name__}: {exc}"))

    def write(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        self.wfile.flush()


def dispatch(state: NodeState, request: dict[str, Any], server: socketserver.BaseServer) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type == "ping":
        return {"ok": True, "type": "pong", "node_id": state.node_id, "monotonic_ms": time.monotonic_ns() // 1_000_000}
    if request_type == "aggregate_batch":
        for field in ("shard_id", "quota_version", "batch_index", "batch_count", "rows"):
            required(request, field)
        started_ns = time.monotonic_ns()
        delay_ms = int(request.get("processing_delay_ms", 0))
        if delay_ms:
            time.sleep(min(30_000, delay_ms) / 1000.0)
        partial = aggregate_batch(request["rows"], state.scale)
        return {
            "ok": True,
            "type": "batch_result",
            "node_id": state.node_id,
            "shard_id": request["shard_id"],
            "quota_version": int(request["quota_version"]),
            "batch_index": int(request["batch_index"]),
            "batch_count": int(request["batch_count"]),
            "caliber_version": AGGREGATION_CALIBER_VERSION,
            "fixed_point_scale": state.scale,
            "partial": partial,
            "node_monotonic_ms": time.monotonic_ns() // 1_000_000,
            "duration_monotonic_ms": max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
        }
    if request_type == "lease_grant":
        return state.grant(request)
    if request_type == "lease_renew":
        return state.renew(request)
    if request_type == "lease_release":
        return state.release(request)
    if request_type == "lease_revoke_rollback":
        return state.release(request, revoked=True)
    if request_type == "quota_set":
        version = int(required(request, "quota_version"))
        with state.lock:
            state.quota_version = version
        return {"ok": True, "node_id": state.node_id, "quota_version": version}
    if request_type == "shutdown":
        threading.Thread(target=server.shutdown, daemon=True).start()
        return {"ok": True, "shutting_down": True}
    return error("E_FIELD", f"unknown request type {request_type!r}", field="type")


class ReusableServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--quota-version", required=True, type=int)
    parser.add_argument("--scale", required=True, type=int)
    parser.add_argument("--io-timeout-ms", type=int, default=5_000)
    args = parser.parse_args(argv)
    NodeHandler.timeout = max(0.05, args.io_timeout_ms / 1000.0)
    state = NodeState(args.node_id, args.seed, args.quota_version, args.scale)
    try:
        server = ReusableServer(("127.0.0.1", args.port), NodeHandler)
    except OSError as exc:
        print(json.dumps({
            "type": "startup_error",
            "code": "E_PORT",
            "field": "port",
            "object_id": args.node_id,
            "message": str(exc),
            "port": args.port,
        }), flush=True)
        return 2
    server.state = state  # type: ignore[attr-defined]
    print(json.dumps({"type": "ready", "node_id": args.node_id, "port": args.port}), flush=True)
    try:
        server.serve_forever(poll_interval=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

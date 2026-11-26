from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import PortInUseError, WaterLeaseError
from .hashing import sha256_hex
from .protocol import PROTOCOL_NAME, PROTOCOL_VERSION, encode_message
from .sampling import aggregate_samples
from .timeutil import elapsed_ms, monotonic_ns


@dataclass
class NodeContext:
    node_id: str
    host: str
    port: int
    seed: int
    fixed_point_scale: int = 100
    processing_delay_ms: int = 0
    request_timeout_ms: int = 2000
    leases: dict[str, dict[str, Any]] = field(default_factory=dict)
    lease_lock: threading.Lock = field(default_factory=threading.Lock)


class JsonLineHandler(socketserver.StreamRequestHandler):
    server: "NodeRpcServer"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.context.request_timeout_ms / 1000.0)

    def handle(self) -> None:
        while True:
            try:
                line = self.rfile.readline()
            except (TimeoutError, OSError):
                return
            if not line:
                return
            message: dict[str, Any] = {}
            try:
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError("message must be an object")
                response = self.server.context.dispatch(message)
            except WaterLeaseError as exc:
                response = {"ok": False, "error": exc.to_dict()}
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                response = {
                    "ok": False,
                    "error": {"code": "E_SPEC", "message": str(exc), "path": "$"},
                }
            try:
                self.wfile.write(encode_message(response))
                self.wfile.flush()
            except (BrokenPipeError, OSError):
                return
            if message.get("type") == "hello" and response.get("ok") is not True:
                return


class NodeRpcServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, context: NodeContext) -> None:
        self.context = context
        try:
            super().__init__((context.host, context.port), JsonLineHandler)
        except OSError as exc:
            raise PortInUseError(
                f"node {context.node_id!r} cannot bind {context.host}:{context.port}: {exc}",
                path="$.nodes.port",
                details={"node_id": context.node_id, "host": context.host, "port": context.port},
            ) from exc


def _error(code: str, message: str, path: str = "$", **details: Any) -> WaterLeaseError:
    return WaterLeaseError(message, path=path, details=details, code_override=code)


def dispatch(self: NodeContext, message: dict[str, Any]) -> dict[str, Any]:
    message_type = message.get("type")
    if message_type == "hello":
        requested_node = message.get("node_id")
        if requested_node is not None and requested_node != self.node_id:
            raise _error(
                "E_NODE",
                "handshake node_id does not match worker",
                "$.node_id",
                expected=self.node_id,
                received=requested_node,
            )
        return {
            "ok": True,
            "type": "hello_ack",
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "node_id": self.node_id,
            "port": self.port,
        }

    if message_type == "aggregate":
        return _handle_aggregate(self, message)
    if message_type == "lease_register":
        return _handle_lease_register(self, message)
    if message_type == "lease_renew":
        return _handle_lease_renew(self, message)
    if message_type == "lease_release":
        return _handle_lease_release(self, message)
    if message_type == "ping":
        return {"ok": True, "type": "pong", "node_id": self.node_id, "monotonic_ns": monotonic_ns()}
    raise _error("E_SPEC", f"unknown message type {message_type!r}", "$.type")


NodeContext.dispatch = dispatch


def _validate_binding(self: NodeContext, message: dict[str, Any]) -> tuple[str, str, str]:
    node_id = message.get("node_id")
    shard_id = message.get("shard_id")
    capacity_version = message.get("capacity_version")
    if node_id != self.node_id:
        raise _error(
            "E_NODE",
            "request node_id does not match worker",
            "$.node_id",
            expected=self.node_id,
            received=node_id,
        )
    if not isinstance(shard_id, str) or not shard_id:
        raise _error("E_SPEC", "shard_id must be a non-empty string", "$.shard_id")
    if not isinstance(capacity_version, str) or not capacity_version:
        raise _error("E_SPEC", "capacity_version must be a non-empty string", "$.capacity_version")
    return node_id, shard_id, capacity_version


def _handle_aggregate(self: NodeContext, message: dict[str, Any]) -> dict[str, Any]:
    _, shard_id, capacity_version = _validate_binding(self, message)
    samples = message.get("samples")
    if not isinstance(samples, list):
        raise _error("E_SPEC", "samples must be a list", "$.samples")
    start = monotonic_ns()
    if self.processing_delay_ms:
        time.sleep(self.processing_delay_ms / 1000.0)
    aggregate = aggregate_samples(samples, fixed_point_scale=self.fixed_point_scale)
    elapsed = elapsed_ms(start)
    aggregate_digest = sha256_hex(aggregate)
    return {
        "ok": True,
        "type": "aggregate_ack",
        "node_id": self.node_id,
        "shard_id": shard_id,
        "capacity_version": capacity_version,
        "aggregate": aggregate,
        "aggregate_digest": aggregate_digest,
        "elapsed_ms": elapsed,
        "monotonic_ns": start,
        "worker_pid": __import__("os").getpid(),
    }


def _handle_lease_register(self: NodeContext, message: dict[str, Any]) -> dict[str, Any]:
    _validate_binding(self, message)
    lease_id = message.get("lease_id")
    if not isinstance(lease_id, str) or not lease_id:
        raise _error("E_LEASE", "lease_id required", "$.lease_id")
    with self.lease_lock:
        existing = self.leases.get(lease_id)
        if existing is not None:
            return {
                "ok": True,
                "type": "lease_register_ack",
                "lease_id": lease_id,
                "lease_version": existing["version"],
                "recovered": True,
            }
        record = {
            "lease_id": lease_id,
            "shard_id": message["shard_id"],
            "node_id": self.node_id,
            "version": 1,
            "renew_count": 0,
        }
        self.leases[lease_id] = record
    return {"ok": True, "type": "lease_register_ack", "lease_id": lease_id, "lease_version": 1}


def _handle_lease_renew(self: NodeContext, message: dict[str, Any]) -> dict[str, Any]:
    _validate_binding(self, message)
    lease_id = message.get("lease_id")
    expected = message.get("expected_version")
    if not isinstance(lease_id, str) or not lease_id:
        raise _error("E_LEASE", "lease_id required", "$.lease_id")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        raise _error("E_LEASE", "expected_version must be a positive integer", "$.expected_version")
    with self.lease_lock:
        record = self.leases.get(lease_id)
        if record is None:
            raise _error(
                "E_LEASE",
                "lease is absent on node",
                "$.lease_id",
                lease_id=lease_id,
                reason="absent",
            )
        if record["version"] != expected:
            raise _error(
                "E_LEASE",
                "optimistic lease version mismatch; expiry was not extended",
                "$.expected_version",
                lease_id=lease_id,
                expected_version=expected,
                actual_version=record["version"],
            )
        record["version"] += 1
        record["renew_count"] += 1
        return {
            "ok": True,
            "type": "lease_renew_ack",
            "lease_id": lease_id,
            "lease_version": record["version"],
            "renew_count": record["renew_count"],
        }


def _handle_lease_release(self: NodeContext, message: dict[str, Any]) -> dict[str, Any]:
    _validate_binding(self, message)
    lease_id = message.get("lease_id")
    expected = message.get("expected_version")
    if not isinstance(lease_id, str) or not lease_id:
        raise _error("E_LEASE", "lease_id required", "$.lease_id")
    with self.lease_lock:
        record = self.leases.pop(lease_id, None)
        if (
            record is not None
            and isinstance(expected, int)
            and not isinstance(expected, bool)
            and record["version"] != expected
        ):
            return {
                "ok": False,
                "error": {
                    "code": "E_LEASE",
                    "message": "lease release version mismatch",
                    "path": "$.expected_version",
                    "details": {"actual_version": record["version"]},
                },
            }
    return {"ok": True, "type": "lease_release_ack", "lease_id": lease_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WaterLease shard node worker")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fixed-point-scale", type=int, default=100)
    parser.add_argument("--processing-delay-ms", type=int, default=0)
    parser.add_argument("--request-timeout-ms", type=int, default=2000)
    args = parser.parse_args(argv)
    context = NodeContext(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        seed=args.seed,
        fixed_point_scale=args.fixed_point_scale,
        processing_delay_ms=args.processing_delay_ms,
        request_timeout_ms=args.request_timeout_ms,
    )
    try:
        server = NodeRpcServer(context)
    except PortInUseError as exc:
        print(json.dumps({"type": "node_start_error", "error": exc.to_dict()}), flush=True)
        return 2
    print(
        json.dumps(
            {
                "type": "node_listening",
                "node_id": context.node_id,
                "host": context.host,
                "port": context.port,
                "pid": __import__("os").getpid(),
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

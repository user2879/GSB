from __future__ import annotations

import argparse
import json
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .samples import make_batch_summary
from .spec import load_spec


class NodeLeaseRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._leases: dict[str, dict[str, Any]] = {}

    def sync(self, lease: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(lease["lease_id"])
        record = {
            "lease_id": lease_id,
            "shard_id": str(lease["shard_id"]),
            "lease_version": int(lease["lease_version"]),
            "renew_count": int(lease.get("renew_count", 0)),
            "expires_at_ms": int(lease.get("expires_at_ms", 0)),
        }
        with self._lock:
            self._leases[lease_id] = record
            return dict(record)

    def release(self, lease_id: str) -> bool:
        with self._lock:
            return self._leases.pop(lease_id, None) is not None

    def renew(
        self,
        *,
        lease_id: str,
        expected_version: int,
        expires_at_ms: int,
    ) -> dict[str, Any]:
        with self._lock:
            current = self._leases.get(lease_id)
            if current is None:
                raise LeaseRejected("lease unknown to node", current_version=None)
            if int(current["lease_version"]) != int(expected_version):
                raise LeaseRejected(
                    "lease version mismatch",
                    current_version=int(current["lease_version"]),
                )
            current["lease_version"] = int(expected_version) + 1
            current["renew_count"] = int(current["renew_count"]) + 1
            current["expires_at_ms"] = int(expires_at_ms)
            return dict(current)


class LeaseRejected(Exception):
    def __init__(self, message: str, *, current_version: int | None) -> None:
        super().__init__(message)
        self.current_version = current_version


REGISTRY = NodeLeaseRegistry()
NODE_CONFIG: dict[str, Any] = {}


class NodeLineHandler(socketserver.StreamRequestHandler):
    def _send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _error(
        self,
        request_id: Any,
        code: str,
        message: str,
        *,
        field: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "ok": False,
            "code": code,
            "message": message,
            "request_id": request_id,
            "node_id": NODE_CONFIG.get("node_id"),
        }
        if field is not None:
            payload["field"] = field
        if extra:
            payload.update(extra)
        self._send(payload)

    def handle(self) -> None:
        handshake_seen = False
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(None, "E_FIELD", "invalid JSON line", field="$")
                return
            if not isinstance(message, dict):
                self._error(None, "E_FIELD", "message must be object", field="$")
                continue

            message_type = message.get("type")
            request_id = message.get("request_id")
            if message_type == "hello":
                if message.get("node_id") != NODE_CONFIG.get("node_id"):
                    self._error(
                        request_id,
                        "E_FIELD",
                        "handshake node_id mismatch",
                        field="node_id",
                    )
                    continue
                handshake_seen = True
                self._send(
                    {
                        "ok": True,
                        "type": "hello_ack",
                        "protocol_version": 1,
                        "node_id": NODE_CONFIG.get("node_id"),
                        "server_time_ms": time.monotonic_ns() // 1_000_000,
                    }
                )
                continue

            if not handshake_seen:
                self._error(request_id, "E_FIELD", "handshake required", field="type")
                continue

            if message_type == "lease_sync":
                self._handle_lease_sync(message, request_id)
            elif message_type == "lease_renew":
                self._handle_lease_renew(message, request_id)
            elif message_type == "lease_release":
                self._handle_lease_release(message, request_id)
            elif message_type == "aggregate_batch":
                self._handle_aggregate(message, request_id)
            elif message_type == "ping":
                self._send(
                    {
                        "ok": True,
                        "type": "pong",
                        "request_id": request_id,
                        "node_id": NODE_CONFIG.get("node_id"),
                        "elapsed_ms_monotonic": 0,
                    }
                )
            else:
                self._error(request_id, "E_FIELD", "unknown message type", field="type")

    def _handle_lease_sync(self, message: dict[str, Any], request_id: Any) -> None:
        for field in ("lease_id", "shard_id", "lease_version", "expires_at_ms"):
            if field not in message:
                self._error(request_id, "E_FIELD", "missing lease field", field=field)
                return
        record = REGISTRY.sync(message)
        self._send(
            {
                "ok": True,
                "type": "lease_synced",
                "request_id": request_id,
                "node_id": NODE_CONFIG["node_id"],
                "shard_id": record["shard_id"],
                "lease_id": record["lease_id"],
                "lease_version": record["lease_version"],
                "renew_count": record["renew_count"],
                "expires_at_ms": record["expires_at_ms"],
                "capacity_version": NODE_CONFIG["capacity_version"],
                "elapsed_ms_monotonic": 0,
            }
        )

    def _handle_lease_renew(self, message: dict[str, Any], request_id: Any) -> None:
        for field in ("lease_id", "lease_version", "expires_at_ms", "shard_id"):
            if field not in message:
                self._error(request_id, "E_FIELD", "missing renew field", field=field)
                return
        try:
            record = REGISTRY.renew(
                lease_id=str(message["lease_id"]),
                expected_version=int(message["lease_version"]),
                expires_at_ms=int(message["expires_at_ms"]),
            )
        except LeaseRejected as exc:
            self._error(
                request_id,
                "E_LEASE",
                str(exc),
                field="lease_version",
                extra={"current_version": exc.current_version},
            )
            return
        self._send(
            {
                "ok": True,
                "type": "lease_renewed",
                "request_id": request_id,
                "node_id": NODE_CONFIG["node_id"],
                "shard_id": message["shard_id"],
                "lease_id": record["lease_id"],
                "lease_version": record["lease_version"],
                "renew_count": record["renew_count"],
                "expires_at_ms": record["expires_at_ms"],
                "capacity_version": NODE_CONFIG["capacity_version"],
                "elapsed_ms_monotonic": 0,
            }
        )

    def _handle_lease_release(self, message: dict[str, Any], request_id: Any) -> None:
        lease_id = message.get("lease_id")
        if not isinstance(lease_id, str):
            self._error(request_id, "E_FIELD", "missing lease_id", field="lease_id")
            return
        existed = REGISTRY.release(lease_id)
        self._send(
            {
                "ok": True,
                "type": "lease_released",
                "request_id": request_id,
                "node_id": NODE_CONFIG["node_id"],
                "shard_id": message.get("shard_id"),
                "lease_id": lease_id,
                "existed": existed,
                "capacity_version": NODE_CONFIG["capacity_version"],
                "elapsed_ms_monotonic": 0,
            }
        )

    def _handle_aggregate(self, message: dict[str, Any], request_id: Any) -> None:
        required = ("shard_id", "capacity_version", "samples")
        for field in required:
            if field not in message:
                self._error(request_id, "E_FIELD", "missing aggregate field", field=field)
                return
        started = time.monotonic_ns()
        try:
            summary = make_batch_summary(
                node_id=str(NODE_CONFIG["node_id"]),
                shard_id=str(message["shard_id"]),
                capacity_version=int(message["capacity_version"]),
                samples=message["samples"],
                fixed_point_exponent=int(NODE_CONFIG.get("fixed_point_exponent", 0)),
                batch_index=int(message.get("batch_index", 0)),
            )
        except Exception as exc:  # noqa: BLE001 - report malformed batch as field/spec data
            self._error(request_id, "E_FIELD", f"invalid samples: {exc}", field="samples")
            return
        elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        self._send(
            {
                "ok": True,
                "type": "aggregate_receipt",
                "request_id": request_id,
                "node_id": NODE_CONFIG["node_id"],
                "shard_id": message["shard_id"],
                "capacity_version": int(message["capacity_version"]),
                "elapsed_ms_monotonic": elapsed_ms,
                "summary": summary,
            }
        )


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def configure(spec_path: str, node_id: str) -> dict[str, Any]:
    spec = load_spec(spec_path)
    if node_id not in spec.node_by_id:
        raise SystemExit(f"unknown node_id: {node_id}")
    return spec.node_by_id[node_id]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Workbench loopback node subprocess")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args(argv)

    global NODE_CONFIG
    NODE_CONFIG = configure(args.spec, args.node_id)
    host = str(NODE_CONFIG["host"])
    port = int(NODE_CONFIG["port"])
    try:
        server = ReusableThreadingTCPServer((host, port), NodeLineHandler)
    except OSError as exc:
        print(
            json.dumps(
                {
                    "code": "E_PORT",
                    "node_id": args.node_id,
                    "host": host,
                    "port": port,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    with server:
        try:
            server.serve_forever(poll_interval=0.05)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

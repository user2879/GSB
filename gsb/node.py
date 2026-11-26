from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

from .aggregate import timed_aggregate
from .clock import monotonic_ms
from .errors import GSBError, LeaseError
from .protocol import recv_line, send_message
from .spec import load_spec


class NodeLeaseState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._leases: dict[str, dict[str, Any]] = {}

    def grant(self, lease: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._leases[lease["lease_id"]] = {
                "lease_id": lease["lease_id"],
                "shard_id": lease["shard_id"],
                "lease_version": int(lease.get("lease_version", 1)),
                "expires_monotonic_ms": int(lease["expires_monotonic_ms"]),
            }
            return self._leases[lease["lease_id"]]

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int | None = None) -> dict[str, Any]:
        with self._lock:
            record = self._leases.get(lease_id)
            if record is None:
                raise LeaseError(f"node does not know lease {lease_id}", field="lease_id", position=lease_id)
            if int(record["lease_version"]) != int(expected_version):
                from .errors import LeaseError

                raise LeaseError(
                    "node optimistic lease version mismatch; expiry not extended",
                    field="lease_version",
                    position=lease_id,
                    expected_version=record["lease_version"],
                    presented_version=expected_version,
                )
            record["lease_version"] += 1
            if ttl_ms is not None:
                record["expires_monotonic_ms"] = monotonic_ms() + int(ttl_ms)
            return record


class NodeHandler:
    def __init__(self, node: dict[str, Any]) -> None:
        self.node = node
        self.leases = NodeLeaseState()

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        message_type = message.get("type")
        if message_type == "hello":
            return {
                "type": "hello_ack",
                "ok": True,
                "node_id": self.node["node_id"],
                "host": "127.0.0.1",
                "port": self.node["port"],
                "protocol": "json-lines-1",
                "monotonic_ms": monotonic_ms(),
            }
        if message_type == "lease_grant":
            record = self.leases.grant(message["lease"])
            return {"ok": True, "type": "lease_grant_ack", "node_id": self.node["node_id"], "lease": record}
        if message_type == "lease_renew":
            record = self.leases.renew(
                message["lease_id"],
                int(message["expected_version"]),
                message.get("ttl_ms"),
            )
            return {"ok": True, "type": "lease_renew_ack", "node_id": self.node["node_id"], "lease": record}
        if message_type in {"aggregate", "process"}:
            return self.aggregate(message)
        raise GSBError(f"unknown protocol message type {message_type!r}")

    def aggregate(self, message: dict[str, Any]) -> dict[str, Any]:
        shard_id = str(message["shard_id"])
        capacity_version = int(self.node["capacity_version"])
        delay_ms = int(self.node.get("processing_delay_ms", 0))
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        summary, elapsed_ms = timed_aggregate(
            message.get("batch_samples", message.get("samples", [])),
            int(self.node["fixed_scale"]),
        )
        return {
            "ok": True,
            "type": "aggregate_result",
            "node_id": self.node["node_id"],
            "shard_id": shard_id,
            "capacity_version": capacity_version,
            "summary": summary,
            "summary_digest": summary["summary_digest"],
            "node_elapsed_ms": elapsed_ms,
            "monotonic_ms": monotonic_ms(),
        }


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _ConnectionHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:  # noqa: A003 - socketserver API
        handler: NodeHandler = self.server.node_handler  # type: ignore[attr-defined]
        try:
            message = recv_line(self.connection)
            try:
                response = handler.handle(message)
            except GSBError as exc:
                response = {"ok": False, **exc.to_dict()}
            send_message(self.connection, response)
        except Exception as exc:  # pragma: no cover - defensive protocol boundary
            response = {"ok": False, "code": "E_PROTOCOL", "error": str(exc)}
            try:
                send_message(self.connection, response)
            except OSError:
                pass


def serve_node(spec_path: str | Path, node_id: str) -> None:
    spec = load_spec(spec_path)
    node = spec["node_by_id"][node_id]
    handler = NodeHandler(node)
    try:
        server = _ThreadingTCPServer(("127.0.0.1", int(node["port"])), _ConnectionHandler)
    except OSError as exc:
        error = {
            "ok": False,
            "code": "E_PORT",
            "error": f"node {node_id} cannot bind 127.0.0.1:{node['port']}: {exc}",
            "field": "nodes.port",
            "position": node["port"],
        }
        print(json.dumps(error, sort_keys=True), flush=True)
        raise SystemExit(2) from exc
    server.node_handler = handler  # type: ignore[attr-defined]
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSB aggregate node subprocess")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args(argv)
    serve_node(args.spec, args.node_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

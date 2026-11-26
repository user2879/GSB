"""Node worker process: real loopback TCP server and lease validation."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading

import numpy as np

from .clock import now_ms
from .protocol import HANDSHAKE_HELLO, HANDSHAKE_READY, read_line, write_line
from .sample import aggregate_rows, merge_aggregate


class NodeServer:
    def __init__(self, node_id: str, host: str, port: int, seed: int, ttl_ms: int):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.seed = seed
        self.ttl_ms = ttl_ms
        self._lock = threading.Lock()
        self._leases: dict[str, dict] = {}
        self._server: socket.socket | None = None

    def lease_snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {key: dict(value) for key, value in self._leases.items()}

    def install_lease(self, lease: dict) -> None:
        with self._lock:
            self._leases[lease["lease_id"]] = dict(lease)

    def revoke_leases_for_version(self, capacity_version: int) -> list[str]:
        with self._lock:
            revoked = [
                lease_id
                for lease_id, lease in self._leases.items()
                if lease["capacity_version"] < capacity_version
            ]
            for lease_id in revoked:
                self._leases.pop(lease_id, None)
            return revoked

    def _validate_dispatch(self, request: dict) -> dict:
        with self._lock:
            lease = self._leases.get(request["lease_id"])
            if lease is None:
                return {"ok": False, "reason": "unknown_lease"}
            if lease["lease_version"] != request["lease_version"]:
                return {"ok": False, "reason": "stale_lease_version"}
            if lease["shard_id"] != request["shard_id"]:
                return {"ok": False, "reason": "shard_mismatch"}
            if lease["capacity_version"] != request["capacity_version"]:
                return {"ok": False, "reason": "stale_capacity_version"}
            if now_ms() >= lease["expire_ms"]:
                self._leases.pop(request["lease_id"], None)
                return {"ok": False, "reason": "lease_expired"}
            return {"ok": True, "lease": dict(lease)}

    def _handle_dispatch(self, request: dict) -> dict:
        check = self._validate_dispatch(request)
        if not check["ok"]:
            return {
                "type": "dispatch_receipt",
                "node_id": self.node_id,
                "shard_id": request["shard_id"],
                "lease_id": request["lease_id"],
                "lease_version": request.get("lease_version"),
                "capacity_version": request.get("capacity_version"),
                "accepted": False,
                "error": check["reason"],
                "summary": None,
                "elapsed_ms": 0,
            }
        start = now_ms()
        merged: dict | None = None
        total_rows = 0
        for batch in request["batches"]:
            array = np.asarray(batch, dtype=np.int64)
            total_rows += int(array.shape[0])
            partial = aggregate_rows(array)
            merged = partial if merged is None else merge_aggregate(merged, partial)
        if merged is None:
            merged = {"participating_rows": 0, "groups": {}}
        if total_rows != request["sample_rows"]:
            return {
                "type": "dispatch_receipt",
                "node_id": self.node_id,
                "shard_id": request["shard_id"],
                "lease_id": request["lease_id"],
                "lease_version": request["lease_version"],
                "capacity_version": request["capacity_version"],
                "accepted": False,
                "error": "row_count_mismatch",
                "summary": None,
                "elapsed_ms": now_ms() - start,
            }
        summary = {
            "node_id": self.node_id,
            "shard_id": request["shard_id"],
            "sample_rows": request["sample_rows"],
            "capacity_version": request["capacity_version"],
            "participating_rows": merged["participating_rows"],
            "groups": merged["groups"],
        }
        return {
            "type": "dispatch_receipt",
            "node_id": self.node_id,
            "shard_id": request["shard_id"],
            "lease_id": request["lease_id"],
            "lease_version": request["lease_version"],
            "capacity_version": request["capacity_version"],
            "accepted": True,
            "summary": summary,
            "elapsed_ms": now_ms() - start,
        }

    def _handle_renew(self, request: dict) -> dict:
        lease = None
        with self._lock:
            lease = self._leases.get(request["lease_id"])
            if lease is None:
                accepted = False
                expire_ms = 0
            elif lease["lease_version"] != request["lease_version"]:
                accepted = False
                expire_ms = lease["expire_ms"]
            elif lease["capacity_version"] != request["capacity_version"]:
                accepted = False
                expire_ms = lease["expire_ms"]
            elif now_ms() >= lease["expire_ms"]:
                self._leases.pop(request["lease_id"], None)
                accepted = False
                expire_ms = 0
            else:
                lease["renewals"] += 1
                lease["lease_version"] += 1
                lease["expire_ms"] = now_ms() + self.ttl_ms
                accepted = True
                expire_ms = lease["expire_ms"]
        return {
            "type": "renew_receipt",
            "node_id": self.node_id,
            "shard_id": request["shard_id"],
            "lease_id": request["lease_id"],
            "lease_version": (lease["lease_version"] if lease else request["lease_version"]),
            "capacity_version": request["capacity_version"],
            "accepted": accepted,
            "expire_ms": expire_ms,
        }

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            hello = read_line(conn)
            if hello != HANDSHAKE_HELLO:
                write_line(conn, {"type": "error", "error": "bad_handshake"})
                return
            write_line(conn, HANDSHAKE_READY)
            request = read_line(conn)
            kind = request.get("type")
            if kind == "dispatch":
                response = self._handle_dispatch(request)
            elif kind == "renew":
                response = self._handle_renew(request)
            elif kind == "ping":
                response = {"type": "pong", "node_id": self.node_id}
            elif kind == "install_lease":
                lease_payload = request.get("lease")
                if not isinstance(lease_payload, dict) or "lease_id" not in lease_payload:
                    response = {"type": "error", "error": "bad_lease_payload"}
                else:
                    self.install_lease(lease_payload)
                    response = {"type": "lease_installed", "lease_id": lease_payload["lease_id"]}
            else:
                response = {"type": "error", "error": f"unknown type {kind}"}
            write_line(conn, response)
        except (ConnectionError, BrokenPipeError):
            return
        finally:
            conn.close()

    def serve(self, ready_event: threading.Event | None = None) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as exc:
            if ready_event is not None:
                ready_event.set()
            print(json.dumps({"event": "bind_error", "error": str(exc)}), file=sys.stderr)
            raise
        server.listen(32)
        server.settimeout(0.25)
        self._server = server
        if ready_event is not None:
            ready_event.set()
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                if self._server is None:
                    return
                continue
            except OSError:
                return
            conn.settimeout(5.0)
            thread = threading.Thread(
                target=self._handle_client, args=(conn,), daemon=True
            )
            thread.start()

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a shardcast node worker")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ttl-ms", type=int, required=True)
    args = parser.parse_args(argv)
    server = NodeServer(args.node_id, args.host, args.port, args.seed, args.ttl_ms)
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

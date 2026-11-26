"""Node subprocess: real 127.0.0.1 TCP listener and JSON Lines protocol."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_digest
from .deterministic import aggregate_partial, samples_from_wire
from .errors import DiagnosticError, LeaseError, PortError, ProtocolError
from .protocol import PROTOCOL_VERSION


class NodeServer:
    def __init__(
        self,
        *,
        node_id: str,
        host: str,
        port: int,
        seed: int,
        fixed_point_decimals: int,
        capacity_version: int = 1,
        concurrency_quota: int = 1,
        lease_limit: int = 1,
        max_renewals: int = 3,
    ) -> None:
        if host != "127.0.0.1":
            raise ProtocolError("node host must be literal 127.0.0.1", field="host")
        self.node_id = node_id
        self.host = host
        self.port = int(port)
        self.seed = int(seed)
        self.fixed_point_decimals = int(fixed_point_decimals)
        self.current_capacity_version = int(capacity_version)
        self.concurrency_quota = int(concurrency_quota)
        self.lease_limit = int(lease_limit)
        self.max_renewals = int(max_renewals)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._leases: dict[str, dict[str, Any]] = {}
        self._active_computes = 0

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except OSError as exc:
            server.close()
            raise PortError(
                f"node {self.node_id!r} cannot bind {self.host}:{self.port}: {exc}",
                node_id=self.node_id,
                host=self.host,
                port=self.port,
            ) from exc
        server.listen(64)
        server.settimeout(0.25)
        self._socket = server
        self._thread = threading.Thread(target=self._accept_loop, name=f"node-{self.node_id}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _accept_loop(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, address = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._connection_loop, args=(conn,), name=f"node-conn-{self.node_id}", daemon=True
            )
            thread.start()

    def _write(self, handle: Any, response: dict[str, Any]) -> None:
        handle.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        handle.flush()

    def _connection_loop(self, conn: socket.socket) -> None:
        with conn:
            try:
                handle = conn.makefile("rwb", buffering=0)
            except OSError:
                return
            with handle:
                while not self._stop.is_set():
                    try:
                        line = handle.readline()
                    except OSError:
                        break
                    if not line:
                        break
                    try:
                        request = json.loads(line.decode("utf-8"))
                        if not isinstance(request, dict):
                            raise ProtocolError("request must be a JSON object")
                        response = self.dispatch(request)
                    except DiagnosticError as exc:
                        response = exc.to_dict()
                    except Exception as exc:  # defensive boundary for subprocess
                        response = DiagnosticError("E_PROTO", f"node request failed: {exc}").to_dict()
                    try:
                        self._write(handle, response)
                    except OSError:
                        break

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "hello":
            return {
                "ok": True,
                "type": "hello_ack",
                "protocol": PROTOCOL_VERSION,
                "node_id": self.node_id,
                "capacity_version": self.current_capacity_version,
            }
        if request_type == "compute":
            return self._compute(request)
        if request_type == "lease_grant":
            return self._lease_grant(request)
        if request_type == "lease_renew":
            return self._lease_renew(request)
        if request_type == "lease_release":
            return self._lease_release(request)
        if request_type == "capacity_advance":
            return self._capacity_advance(request)
        raise ProtocolError(f"unknown request type {request_type!r}")

    def _compute(self, request: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        required = ("request_id", "shard_id", "capacity_version", "samples")
        missing = [name for name in required if name not in request]
        if missing:
            raise ProtocolError("compute request missing fields", missing_fields=missing)
        if request.get("node_id") is not None and request.get("node_id") != self.node_id:
            raise ProtocolError("compute node_id mismatch", expected=self.node_id, actual=request.get("node_id"))
        if request.get("seed") is not None and int(request["seed"]) != self.seed:
            raise ProtocolError("compute seed does not match declared node seed")
        capacity_version = int(request["capacity_version"])
        if capacity_version != self.current_capacity_version:
            raise LeaseError(
                "compute capacity generation mismatch",
                expected_generation=self.current_capacity_version,
                basis_generation=self.current_capacity_version,
                actual_generation=capacity_version,
            )

        with self._lock:
            if self._active_computes >= self.concurrency_quota:
                raise DiagnosticError(
                    "E_QUOTA",
                    "node concurrency quota would be exceeded inside subprocess",
                    node_id=self.node_id,
                    concurrency_quota=self.concurrency_quota,
                )
            self._active_computes += 1
        try:
            samples = samples_from_wire(request["samples"])
            partial = aggregate_partial(
                samples,
                fixed_point_decimals=self.fixed_point_decimals,
                batch_index=int(request.get("batch_index", 0)),
            )
        finally:
            with self._lock:
                self._active_computes -= 1

        elapsed_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        partial_digest = canonical_digest(partial)
        row_count = int(partial["row_count"])
        participating = int(partial["participating_rows"])
        summary = {
            "node_id": self.node_id,
            "shard_id": request["shard_id"],
            "capacity_version": capacity_version,
            "sample_rows": row_count,
            "participating_rows": participating,
            "fixed_point_decimals": self.fixed_point_decimals,
            "partial_digest": partial_digest,
            "summary_digest": partial_digest,
        }
        return {
            "ok": True,
            "request_id": request["request_id"],
            "node_id": self.node_id,
            "shard_id": request["shard_id"],
            "capacity_version": capacity_version,
            "batch_index": int(request.get("batch_index", 0)),
            "partial": partial,
            "summary": summary,
            "elapsed_monotonic_ms": int(elapsed_ms),
        }

    def _lease_grant(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(request["lease_id"])
        with self._lock:
            active = [lease for lease in self._leases.values() if lease["status"] == "active"]
            if lease_id in self._leases:
                lease = self._leases[lease_id]
                return {"ok": True, "lease_id": lease_id, "lease_version": lease["lease_version"], "status": lease["status"]}
            if len(active) >= self.lease_limit:
                raise LeaseError(
                    "node lease_limit would be exceeded",
                    node_id=self.node_id,
                    lease_limit=self.lease_limit,
                    active_leases=len(active),
                )
            lease = {
                "lease_id": lease_id,
                "shard_id": str(request["shard_id"]),
                "node_id": self.node_id,
                "capacity_version": int(request.get("capacity_version", self.current_capacity_version)),
                "granted_monotonic_ms": int(request["granted_monotonic_ms"]),
                "expires_monotonic_ms": int(request["expires_monotonic_ms"]),
                "ttl_ms": int(request.get("ttl_ms", 0)),
                "renew_count": 0,
                "lease_version": 1,
                "status": "active",
            }
            self._leases[lease_id] = lease
            return {
                "ok": True,
                "lease_id": lease_id,
                "lease_version": 1,
                "status": "active",
                "expires_monotonic_ms": lease["expires_monotonic_ms"],
            }

    def _lease_renew(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(request["lease_id"])
        expected_version = int(request["expected_version"])
        max_renewals = int(request.get("max_renewals", self.max_renewals))
        ttl_ms = int(request.get("ttl_ms", 0))
        now_ms = time.monotonic_ns() // 1_000_000
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease["status"] != "active":
                raise LeaseError(
                    "lease is unknown, released, or revoked by capacity generation",
                    lease_id=lease_id,
                    basis_generation=self.current_capacity_version,
                    capacity_generation=self.current_capacity_version,
                )
            if lease["capacity_version"] != self.current_capacity_version:
                raise LeaseError(
                    "lease belongs to a stale capacity generation",
                    lease_id=lease_id,
                    expected_version=expected_version,
                    actual_version=lease["lease_version"],
                    basis_generation=lease["lease_version"],
                    lease_generation=lease["capacity_version"],
                    capacity_generation=self.current_capacity_version,
                )
            if lease["lease_version"] != expected_version:
                raise LeaseError(
                    "optimistic lease version mismatch; expiry was not extended",
                    lease_id=lease_id,
                    expected_version=expected_version,
                    actual_version=lease["lease_version"],
                    basis_generation=lease["lease_version"],
                    capacity_generation=self.current_capacity_version,
                )
            if lease["renew_count"] >= max_renewals:
                raise LeaseError(
                    "lease renewal count exceeds declared limit",
                    lease_id=lease_id,
                    max_renewals=max_renewals,
                    renew_count=lease["renew_count"],
                    basis_generation=lease["lease_version"],
                )
            if now_ms >= int(lease["expires_monotonic_ms"]):
                lease["status"] = "expired"
                raise LeaseError(
                    "lease has already expired; renewal is rejected",
                    lease_id=lease_id,
                    basis_generation=lease["lease_version"],
                    expires_monotonic_ms=lease["expires_monotonic_ms"],
                    now_monotonic_ms=now_ms,
                )
            lease["renew_count"] += 1
            lease["lease_version"] += 1
            if ttl_ms > 0:
                lease["expires_monotonic_ms"] = now_ms + ttl_ms
            return {
                "ok": True,
                "lease_id": lease_id,
                "lease_version": lease["lease_version"],
                "renew_count": lease["renew_count"],
                "status": "active",
                "expires_monotonic_ms": lease["expires_monotonic_ms"],
                "basis_generation": lease["lease_version"],
            }

    def _lease_release(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(request["lease_id"])
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is not None:
                lease["status"] = "released"
            return {"ok": True, "lease_id": lease_id, "status": "released" if lease else "not_found"}

    def _capacity_advance(self, request: dict[str, Any]) -> dict[str, Any]:
        to_version = int(request["to_version"])
        parent_version = request.get("parent_version")
        with self._lock:
            if parent_version is not None and int(parent_version) != self.current_capacity_version:
                raise LeaseError(
                    "capacity parent generation does not match node generation",
                    expected_parent=self.current_capacity_version,
                    actual_parent=int(parent_version),
                    basis_generation=self.current_capacity_version,
                )
            revoked: list[str] = []
            for lease_id, lease in list(self._leases.items()):
                if int(lease["capacity_version"]) < to_version and lease["status"] == "active":
                    lease["status"] = "revoked_capacity_generation"
                    revoked.append(lease_id)
            self.current_capacity_version = to_version
            if "concurrency_quota" in request:
                self.concurrency_quota = int(request["concurrency_quota"])
            if "lease_limit" in request:
                self.lease_limit = int(request["lease_limit"])
            if "fixed_point_decimals" in request:
                self.fixed_point_decimals = int(request["fixed_point_decimals"])
            return {
                "ok": True,
                "status": "advanced",
                "current_capacity_version": to_version,
                "revoked_lease_ids": sorted(revoked),
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shard diagnostic TCP node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fixed-point-decimals", type=int, default=0)
    parser.add_argument("--capacity-version", type=int, default=1)
    parser.add_argument("--concurrency-quota", type=int, default=1)
    parser.add_argument("--lease-limit", type=int, default=1)
    parser.add_argument("--max-renewals", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = NodeServer(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        seed=args.seed,
        fixed_point_decimals=args.fixed_point_decimals,
        capacity_version=args.capacity_version,
        concurrency_quota=args.concurrency_quota,
        lease_limit=args.lease_limit,
        max_renewals=args.max_renewals,
    )
    try:
        server.start()
    except DiagnosticError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps({"ok": True, "type": "node_started", "node_id": args.node_id, "port": args.port}),
        flush=True,
    )
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

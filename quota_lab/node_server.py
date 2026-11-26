from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregation import aggregate_samples
from .errors import QuotaLabError
from .spec import SessionSpec, load_session_files


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class LeaseRecord:
    lease_id: str
    shard_id: str
    lease_version: int
    renew_count: int
    capacity_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "lease_version": self.lease_version,
            "renew_count": self.renew_count,
            "capacity_version": self.capacity_version,
        }


class NodeBackend:
    def __init__(self, spec: SessionSpec, node_id: str) -> None:
        self.spec = spec
        self.node = spec.node(node_id)
        self.current_version = int(self.node.capacity_version)
        self.leases: dict[str, LeaseRecord] = {}
        self.lock = threading.RLock()

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        op = payload.get("op")
        if op == "hello":
            return {
                "ok": True,
                "op": "hello",
                "node_id": self.node.node_id,
                "host": self.node.host,
                "port": self.node.port,
                "capacity_version": self.current_version,
                "pid": __import__("os").getpid(),
                "monotonic_ms": monotonic_ms(),
            }
        if payload.get("node_id") != self.node.node_id:
            return self._error("E_SPEC", "node_id does not match child process", "node_id")
        if op == "aggregate":
            return self._aggregate(payload)
        if op == "lease_grant":
            return self._grant(payload)
        if op == "lease_renew":
            return self._renew(payload)
        if op == "lease_revoke":
            return self._revoke(payload)
        if op == "advance_version":
            return self._advance_version(payload)
        return self._error("E_SPEC", f"unknown op {op!r}", "op")

    def _error(self, code: str, message: str, field: str | None = None, details: dict | None = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if field:
            error["first_error_field"] = field
        if details:
            error["details"] = details
        return {"ok": False, "error": error}

    def _aggregate(self, payload: dict[str, Any]) -> dict[str, Any]:
        shard_id = payload.get("shard_id")
        if not isinstance(shard_id, str):
            return self._error("E_SPEC", "shard_id must be string", "shard_id")
        try:
            shard = self.spec.shard(shard_id)
        except KeyError:
            return self._error("E_SPEC", "unknown shard_id", "shard_id")
        if shard.node_id != self.node.node_id:
            return self._error("E_SPEC", "shard is not assigned to this node", "shard_id")

        requested_version = payload.get("capacity_version")
        if requested_version != self.current_version:
            return self._error(
                "E_VERSION",
                "capacity version does not match active node version",
                "capacity_version",
                {"requested": requested_version, "active": self.current_version},
            )

        samples = payload.get("samples")
        if not isinstance(samples, list):
            return self._error("E_SPEC", "samples must be a list", "samples")

        delay_ms = self.spec.fault_injection.delay_ms_by_shard.get(shard_id, 0)
        if self.spec.fault_injection.timeout_shard_id == shard_id:
            delay_ms = max(delay_ms, self.spec.fault_injection.timeout_delay_ms)
        request_delay = payload.get("fault_delay_ms", 0)
        if isinstance(request_delay, int) and not isinstance(request_delay, bool) and request_delay > 0:
            delay_ms = max(delay_ms, request_delay)
        if delay_ms:
            time.sleep(delay_ms / 1000.0)

        started = time.monotonic_ns()
        try:
            summary = aggregate_samples(
                samples,
                q=self.spec.q,
                pressure_den=self.spec.accounting.pressure_den,
            )
        except Exception as exc:  # deterministic local input validation
            return self._error("E_SPEC", f"aggregation failed: {exc}", "samples")
        elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
        return {
            "ok": True,
            "op": "aggregate",
            "node_id": self.node.node_id,
            "shard_id": shard_id,
            "capacity_version": self.current_version,
            "summary": summary,
            "elapsed_ms": int(elapsed_ms),
            "monotonic_ms": monotonic_ms(),
        }

    def _grant(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = payload.get("lease_id")
        shard_id = payload.get("shard_id")
        if not isinstance(lease_id, str) or not isinstance(shard_id, str):
            return self._error("E_LEASE", "lease_id and shard_id must be strings", "lease_id")
        with self.lock:
            existing = self.leases.get(lease_id)
            if existing is not None:
                if existing.shard_id != shard_id:
                    return self._error("E_LEASE", "lease id reused for another shard", "lease_id")
                return {"ok": True, "lease": existing.to_dict(), "idempotent": True}
            record = LeaseRecord(
                lease_id=lease_id,
                shard_id=shard_id,
                lease_version=1,
                renew_count=0,
                capacity_version=self.current_version,
            )
            self.leases[lease_id] = record
            return {"ok": True, "lease": record.to_dict(), "idempotent": False}

    def _renew(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = payload.get("lease_id")
        expected_version = payload.get("expected_version")
        if not isinstance(lease_id, str) or not isinstance(expected_version, int) or isinstance(expected_version, bool):
            return self._error("E_LEASE", "lease_id and integer expected_version are required", "lease_version")
        with self.lock:
            record = self.leases.get(lease_id)
            if record is None:
                return self._error("E_LEASE", "lease is unknown to node", "lease_id", {"lease_id": lease_id})
            if record.lease_version != expected_version:
                return self._error(
                    "E_LEASE",
                    "optimistic lease version check failed; expiry was not extended",
                    "lease_version",
                    {"expected": expected_version, "current": record.lease_version},
                )
            record.lease_version += 1
            record.renew_count += 1
            return {"ok": True, "lease": record.to_dict()}

    def _revoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease_id = payload.get("lease_id")
        with self.lock:
            existed = self.leases.pop(lease_id, None) if isinstance(lease_id, str) else None
            return {"ok": True, "revoked": existed is not None, "lease_id": lease_id}

    def _advance_version(self, payload: dict[str, Any]) -> dict[str, Any]:
        new_version = payload.get("new_capacity_version")
        with self.lock:
            if new_version is None:
                new_version = self.current_version + 1
            if not isinstance(new_version, int) or isinstance(new_version, bool) or new_version < self.current_version:
                return self._error("E_SPEC", "new_capacity_version must be non-decreasing integer", "capacity_version")
            revoked = [lease.to_dict() for lease in self.leases.values()]
            self.leases.clear()
            self.current_version = new_version
            return {
                "ok": True,
                "node_id": self.node.node_id,
                "capacity_version": self.current_version,
                "revoked_leases": revoked,
            }


class JsonlineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        backend: NodeBackend = self.server.backend  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                payload = json.loads(line.decode("utf-8"))
                if not isinstance(payload, dict):
                    response = {"ok": False, "error": {"code": "E_PROTOCOL", "message": "request must be object"}}
                else:
                    response = backend.handle(payload)
            except Exception as exc:
                response = {"ok": False, "error": {"code": "E_NODE", "message": str(exc)}}
            self.wfile.write((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
            self.wfile.flush()
            if payload.get("op") not in {"hello", "lease_renew", "lease_grant"}:
                # One aggregate/admin operation per connection keeps failure domains simple.
                return


class ThreadingJsonlineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(spec_path: str | Path, node_id: str) -> int:
    declaration_path = Path(spec_path)
    # The child receives the same combined session JSON when generated by the CLI.
    data = json.loads(declaration_path.read_text(encoding="utf-8"))
    if "shards" in data and isinstance(data.get("shards"), list):
        from .spec import parse_session

        spec = parse_session(data, {"shards": data["shards"]})
    else:
        shards_path = data["_shards_path"] if "_shards_path" in data else declaration_path.with_name("shards.json")
        spec = load_session_files(declaration_path, shards_path)
    backend = NodeBackend(spec, node_id)
    try:
        server = ThreadingJsonlineServer((spec.node(node_id).host, spec.node(node_id).port), JsonlineHandler)
    except OSError as exc:
        error = QuotaLabError(
            f"node {node_id} cannot bind {spec.node(node_id).host}:{spec.node(node_id).port}: {exc}",
            field=f"nodes.{node_id}.port",
            code="E_PORT",
        )
        print(json.dumps(error.to_dict(), sort_keys=True), flush=True)
        return 2
    server.backend = backend  # type: ignore[attr-defined]
    with server:
        server.serve_forever(poll_interval=0.05)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quota lab TCP JSON-line node")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args(argv)
    return serve(args.spec, args.node_id)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import os
import socketserver
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json
from .errors import DiagnosticError, ErrorCode
from .protocol import PROTOCOL_VERSION, MAX_LINE_BYTES, decode_line, encode_line
from .routing import generate_sample_keys, sample_hit_shards
from .spec import DatabaseSpec, Spec, TableRule, load_spec


@dataclass
class _WorkerLease:
    lease_id: str
    shard_id: str
    event_seq: int
    version: int
    active: bool = True


class LeaseStore:
    """Child-side optimistic lease version authority."""

    def __init__(self) -> None:
        self._leases: dict[str, _WorkerLease] = {}
        self._lock = threading.Lock()

    def grant(self, lease_id: str, shard_id: str, event_seq: int, version: int) -> int:
        with self._lock:
            existing = self._leases.get(lease_id)
            if existing is not None:
                if existing.shard_id != shard_id or existing.event_seq != event_seq:
                    raise DiagnosticError(
                        ErrorCode.LEASE.value,
                        f"lease {lease_id!r} already exists with a different binding",
                        field="lease_id",
                    )
                if not existing.active:
                    existing.version = version
                    existing.active = True
                    return existing.version
                if existing.version == version:
                    return existing.version
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} is active at version {existing.version}, not {version}",
                    field="lease_id",
                )
            if version < 1:
                raise DiagnosticError(ErrorCode.LEASE.value, "lease version must be positive", field="lease_version")
            self._leases[lease_id] = _WorkerLease(lease_id, shard_id, event_seq, version)
            return version

    def renew(
        self,
        lease_id: str,
        expected_version: int,
        shard_id: str,
        event_seq: int,
    ) -> int:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or not lease.active:
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} is unknown or inactive",
                    field="lease_id",
                )
            if lease.shard_id != shard_id or lease.event_seq != event_seq:
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} binding does not match",
                    field="lease_id",
                )
            if lease.version != expected_version:
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} version {expected_version} is stale; current version is {lease.version}",
                    field="lease_version",
                    details={"expected": expected_version, "actual": lease.version},
                )
            lease.version += 1
            return lease.version

    def revoke(self, lease_id: str) -> int:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise DiagnosticError(ErrorCode.LEASE.value, f"lease {lease_id!r} is unknown", field="lease_id")
            lease.active = False
            return lease.version


class RoutingService:
    def __init__(self, spec: Spec, database: DatabaseSpec) -> None:
        self.spec = spec
        self.database = database
        self.leases = LeaseStore()

    def _rule(self, table: str, route_version: int) -> TableRule:
        if route_version != self.database.route_version:
            raise DiagnosticError(
                ErrorCode.PROTO.value,
                f"probe requested route version {route_version}, worker is bound to {self.database.route_version}",
                db_id=self.database.id,
                field="route_version",
            )
        rule = self.spec.maybe_rule(table, self.database.route_version)
        if rule is None:
            raise DiagnosticError(
                ErrorCode.PROTO.value,
                f"table {table!r} has no rule at worker route version {self.database.route_version}",
                db_id=self.database.id,
                field="table",
            )
        return rule

    def probe_one(self, probe: dict[str, Any], batch_id: str | None) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        event_seq = probe.get("event_seq")
        route_key = probe.get("route_key")
        table = probe.get("table")
        route_version = probe.get("route_version", self.database.route_version)
        sample_count = probe.get("sample_count", 1)
        if not isinstance(event_seq, int) or isinstance(event_seq, bool):
            raise DiagnosticError(ErrorCode.PROTO.value, "probe event_seq must be an integer", field="event_seq")
        if not isinstance(route_key, int) or isinstance(route_key, bool):
            raise DiagnosticError(ErrorCode.PROTO.value, "probe route_key must be an integer", field="route_key")
        if not isinstance(table, str) or not table:
            raise DiagnosticError(ErrorCode.PROTO.value, "probe table must be a non-empty string", field="table")
        if not isinstance(route_version, int) or isinstance(route_version, bool):
            raise DiagnosticError(ErrorCode.PROTO.value, "probe route_version must be an integer", field="route_version")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or not 1 <= sample_count <= 4096:
            raise DiagnosticError(
                ErrorCode.PROTO.value,
                "probe sample_count must be an integer in 1..4096",
                field="sample_count",
            )
        rule = self._rule(table, route_version)
        supplied_samples = probe.get("sample_keys")
        if supplied_samples is not None:
            if not isinstance(supplied_samples, list) or not all(
                isinstance(key, int) and not isinstance(key, bool) for key in supplied_samples
            ):
                raise DiagnosticError(ErrorCode.PROTO.value, "sample_keys must be integers", field="sample_keys")
            samples = tuple(supplied_samples)
        else:
            samples = generate_sample_keys(
                seed=self.database.seed,
                event_seq=event_seq,
                route_key=route_key,
                rule=rule,
                sample_count=sample_count,
            )
        shards = sample_hit_shards(rule, samples)
        elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        return {
            "db_id": self.database.id,
            "event_seq": event_seq,
            "hit_shards": list(shards),
            "route_version": self.database.route_version,
            "elapsed_ms": round(elapsed_ms, 6),
            "sample_keys": list(samples),
            "batch_id": batch_id,
        }

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "ping":
            return {
                "ok": True,
                "type": "pong",
                "protocol": PROTOCOL_VERSION,
                "pid": os.getpid(),
                "database": self.database.as_dict(),
            }
        if request_type == "probe_batch":
            probes = request.get("probes")
            batch_id = request.get("batch_id")
            if not isinstance(probes, list) or not probes or len(probes) > 4096:
                raise DiagnosticError(ErrorCode.PROTO.value, "probes must be a non-empty list", field="probes")
            results = [self.probe_one(probe, batch_id) for probe in probes]
            results.sort(key=lambda result: (result["event_seq"], result["db_id"]))
            return {
                "ok": True,
                "type": "probe_batch_result",
                "batch_id": batch_id,
                "db_id": self.database.id,
                "route_version": self.database.route_version,
                "results": results,
            }
        if request_type == "lease_grant":
            lease_id = request.get("lease_id")
            shard_id = request.get("shard_id")
            event_seq = request.get("event_seq")
            version = request.get("lease_version")
            if not isinstance(lease_id, str) or not isinstance(shard_id, str) or not isinstance(event_seq, int):
                raise DiagnosticError(ErrorCode.PROTO.value, "invalid lease grant fields", field="lease_grant")
            if not isinstance(version, int) or isinstance(version, bool):
                raise DiagnosticError(ErrorCode.PROTO.value, "lease_version must be an integer", field="lease_version")
            granted_version = self.leases.grant(lease_id, shard_id, event_seq, version)
            return {
                "ok": True,
                "type": "lease_granted",
                "lease_id": lease_id,
                "lease_version": granted_version,
            }
        if request_type == "lease_renew":
            lease_id = request.get("lease_id")
            expected = request.get("expected_lease_version")
            shard_id = request.get("shard_id")
            event_seq = request.get("event_seq")
            if not isinstance(lease_id, str) or not isinstance(shard_id, str) or not isinstance(event_seq, int):
                raise DiagnosticError(ErrorCode.PROTO.value, "invalid lease renew fields", field="lease_renew")
            if not isinstance(expected, int) or isinstance(expected, bool):
                raise DiagnosticError(
                    ErrorCode.PROTO.value,
                    "expected_lease_version must be an integer",
                    field="expected_lease_version",
                )
            new_version = self.leases.renew(lease_id, expected, shard_id, event_seq)
            return {
                "ok": True,
                "type": "lease_renewed",
                "lease_id": lease_id,
                "lease_version": new_version,
            }
        if request_type == "lease_revoke":
            lease_id = request.get("lease_id")
            if not isinstance(lease_id, str):
                raise DiagnosticError(ErrorCode.PROTO.value, "lease_id must be a string", field="lease_id")
            version = self.leases.revoke(lease_id)
            return {
                "ok": True,
                "type": "lease_revoked",
                "lease_id": lease_id,
                "lease_version": version,
            }
        raise DiagnosticError(ErrorCode.PROTO.value, f"unknown request type {request_type!r}", field="type")


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RequestHandler(socketserver.StreamRequestHandler):
    server: "_ThreadingTCPServerWithService"

    def _read_payload(self) -> dict[str, Any] | None:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = self.rfile.readline(MAX_LINE_BYTES + 1)
            if not chunk:
                return None
            total += len(chunk)
            if total > MAX_LINE_BYTES:
                raise DiagnosticError(ErrorCode.PROTO.value, "JSON line exceeds protocol size limit", field="line")
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
        return decode_line(b"".join(chunks).strip())

    def _send(self, payload: dict[str, Any]) -> None:
        self.wfile.write(encode_line(payload))
        self.wfile.flush()

    def handle(self) -> None:
        service = self.server.service
        self.request.settimeout(10.0)
        try:
            hello = self._read_payload()
            if hello is None:
                return
            if hello.get("type") != "hello":
                raise DiagnosticError(ErrorCode.PROTO.value, "first line must be a hello handshake", field="type")
            if hello.get("protocol") != PROTOCOL_VERSION:
                raise DiagnosticError(ErrorCode.PROTO.value, "unsupported protocol version", field="protocol")
            if hello.get("db_id") != service.database.id:
                raise DiagnosticError(ErrorCode.PROTO.value, "hello db_id does not match worker", field="db_id")
            self._send(
                {
                    "ok": True,
                    "type": "hello_ack",
                    "protocol": PROTOCOL_VERSION,
                    "db_id": service.database.id,
                    "route_version": service.database.route_version,
                }
            )
            while True:
                request = self._read_payload()
                if request is None:
                    return
                try:
                    self._send(service.dispatch(request))
                except DiagnosticError as error:
                    response = error.to_dict()
                    response.update({"type": "error", "db_id": service.database.id})
                    self._send(response)
        except (BrokenPipeError, ConnectionResetError, DiagnosticError):
            return


class _ThreadingTCPServerWithService(_ThreadingTCPServer):
    def __init__(self, address: tuple[str, int], service: RoutingService) -> None:
        self.service = service
        super().__init__(address, _RequestHandler)


def _emit_startup_error(code: str, message: str, db_id: str) -> None:
    payload = {"ok": False, "type": "fatal", "code": code, "db_id": db_id, "message": message}
    print(canonical_json(payload), flush=True)


def run_worker(spec_path: str, db_id: str) -> int:
    spec = load_spec(spec_path)
    try:
        database = spec.database(db_id)
    except DiagnosticError as error:
        _emit_startup_error(ErrorCode.SPEC.value, error.message, db_id)
        return 2
    service = RoutingService(spec, database)
    try:
        server = _ThreadingTCPServerWithService((database.host, database.port), service)
    except OSError as error:
        _emit_startup_error(ErrorCode.PORT.value, f"cannot bind {database.host}:{database.port}: {error}", db_id)
        return 2
    print(
        canonical_json(
            {
                "ok": True,
                "type": "ready",
                "db_id": database.id,
                "host": database.host,
                "port": database.port,
                "route_version": database.route_version,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="physical shard routing child process")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--db-id", required=True)
    args = parser.parse_args(argv)
    return run_worker(args.spec, args.db_id)


if __name__ == "__main__":
    sys.exit(main())

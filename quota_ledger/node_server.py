"""Deterministic NumPy aggregation node speaking JSON lines over loopback TCP."""
from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .util import canonical_json, digest_obj, monotonic_ms


def _line(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


@dataclass
class LeaseRecord:
    lease_id: str
    shard_id: str
    granted_ms: int
    expires_ms: int
    version: int
    ttl_ms: int
    state: str = "active"  # active | completed | released | expired


@dataclass
class NodeState:
    node_id: str
    seed: int
    concurrency: int
    lease_limit: int
    capacity_version: int
    default_ttl_ms: int
    lock: threading.RLock = field(default_factory=threading.RLock)
    active_computes: int = 0
    leases: dict[str, LeaseRecord] = field(default_factory=dict)

    def _expire_locked(self, now: int) -> None:
        for lease in self.leases.values():
            if lease.state == "active" and now >= lease.expires_ms:
                lease.state = "expired"

    def active_leases_locked(self) -> list[LeaseRecord]:
        self._expire_locked(monotonic_ms())
        return [lease for lease in self.leases.values() if lease.state == "active"]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._expire_locked(monotonic_ms())
            active = sorted(
                (
                    {
                        "lease_id": lease.lease_id,
                        "shard_id": lease.shard_id,
                        "version": lease.version,
                        "granted_ms": lease.granted_ms,
                        "expires_ms": lease.expires_ms,
                        "state": lease.state,
                    }
                    for lease in self.leases.values()
                    if lease.state == "active"
                ),
                key=lambda item: (item["shard_id"], item["lease_id"]),
            )
            return {
                "node_id": self.node_id,
                "concurrency": self.concurrency,
                "lease_limit": self.lease_limit,
                "capacity_version": self.capacity_version,
                "active_computes": self.active_computes,
                "active_leases": active,
            }

    def ping(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "node_id": self.node_id,
                "ready": True,
                "pid": os.getpid(),
                "concurrency": self.concurrency,
                "lease_limit": self.lease_limit,
                "capacity_version": self.capacity_version,
            }

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            active = self.active_leases_locked()
            concurrency = int(request.get("concurrency", self.concurrency))
            lease_limit = int(request.get("lease_limit", self.lease_limit))
            capacity_version = int(request.get("capacity_version", self.capacity_version))
            if concurrency < self.active_computes:
                return {"ok": False, "code": "E_LEASE", "message": "concurrency below active computes"}
            if lease_limit < len(active):
                return {"ok": False, "code": "E_LEASE", "message": "lease_limit below active leases"}
            self.concurrency = concurrency
            self.lease_limit = lease_limit
            self.capacity_version = capacity_version
            return {
                "ok": True,
                "node_id": self.node_id,
                "concurrency": self.concurrency,
                "lease_limit": self.lease_limit,
                "capacity_version": self.capacity_version,
            }

    def acquire_lease(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        shard_id = request.get("shard_id")
        if not isinstance(lease_id, str) or not lease_id or not isinstance(shard_id, str) or not shard_id:
            return {"ok": False, "code": "E_REPLY", "message": "lease_id and shard_id are required"}
        ttl_ms = int(request.get("ttl_ms", self.default_ttl_ms))
        if ttl_ms <= 0:
            return {"ok": False, "code": "E_LEASE", "message": "ttl_ms must be positive"}
        with self.lock:
            self._expire_locked(monotonic_ms())
            active = [lease for lease in self.leases.values() if lease.state == "active"]
            if any(lease.shard_id == shard_id for lease in active):
                return {"ok": False, "code": "E_LEASE", "message": "shard already has an active lease"}
            if any(lease.lease_id == lease_id for lease in active):
                return {"ok": False, "code": "E_LEASE", "message": "lease_id already active"}
            if len(active) >= self.lease_limit:
                return {"ok": False, "code": "E_LEASE", "message": "node lease limit exhausted"}
            now = monotonic_ms()
            lease = LeaseRecord(
                lease_id=lease_id,
                shard_id=shard_id,
                granted_ms=now,
                expires_ms=now + ttl_ms,
                version=1,
                ttl_ms=ttl_ms,
            )
            self.leases[lease_id] = lease
            return {
                "ok": True,
                "node_id": self.node_id,
                "lease_id": lease.lease_id,
                "shard_id": lease.shard_id,
                "granted_ms": lease.granted_ms,
                "expires_ms": lease.expires_ms,
                "version": lease.version,
            }

    def renew_lease(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        expected_version = request.get("expected_version")
        ttl_ms_raw = request.get("ttl_ms")
        if not isinstance(lease_id, str) or not isinstance(expected_version, int):
            return {"ok": False, "code": "E_REPLY", "message": "lease_id and integer expected_version are required"}
        with self.lock:
            self._expire_locked(monotonic_ms())
            lease = self.leases.get(lease_id)
            now = monotonic_ms()
            if lease is None or lease.state != "active" or now >= lease.expires_ms:
                return {"ok": False, "code": "E_LEASE", "message": "lease absent or expired"}
            if lease.version != expected_version:
                return {
                    "ok": False,
                    "code": "E_LEASE",
                    "message": "lease version mismatch",
                    "expected_version": expected_version,
                    "actual_version": lease.version,
                }
            ttl_ms = int(ttl_ms_raw) if ttl_ms_raw is not None else lease.ttl_ms
            if ttl_ms <= 0:
                return {"ok": False, "code": "E_LEASE", "message": "ttl_ms must be positive"}
            lease.version += 1
            lease.ttl_ms = ttl_ms
            lease.expires_ms = now + ttl_ms
            return {
                "ok": True,
                "node_id": self.node_id,
                "lease_id": lease.lease_id,
                "version": lease.version,
                "expires_ms": lease.expires_ms,
            }

    def release_lease(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        with self.lock:
            lease = self.leases.get(str(lease_id)) if lease_id is not None else None
            if lease is not None and lease.state in {"active", "completed"}:
                lease.state = "released"
            return {"ok": True, "node_id": self.node_id, "lease_id": lease_id, "released": lease is not None}

    def compute(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        required = (
            "node_id",
            "shard_id",
            "config_version",
            "lease_id",
            "lease_version",
            "samples",
            "scale",
            "group_modulus",
        )
        for field_name in required:
            if field_name not in request:
                return {"ok": False, "code": "E_REPLY", "message": f"missing field: {field_name}"}
        if request["node_id"] != self.node_id:
            return {"ok": False, "code": "E_REPLY", "message": "node_id mismatch"}
        lease_id = request["lease_id"]
        with self.lock:
            self._expire_locked(monotonic_ms())
            lease = self.leases.get(str(lease_id))
            if lease is None or lease.state != "active":
                return {"ok": False, "code": "E_LEASE", "message": "lease absent or inactive"}
            if lease.shard_id != request["shard_id"]:
                return {"ok": False, "code": "E_LEASE", "message": "lease belongs to another shard"}
            if lease.version != request["lease_version"]:
                return {"ok": False, "code": "E_LEASE", "message": "lease version mismatch"}
            if self.active_computes >= self.concurrency:
                return {"ok": False, "code": "E_LEASE", "message": "node concurrency exhausted"}
            self.active_computes += 1
        try:
            samples = request["samples"]
            if not isinstance(samples, list):
                return {"ok": False, "code": "E_REPLY", "message": "samples must be an array"}
            scale = request["scale"]
            groups = request["group_modulus"]
            if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
                return {"ok": False, "code": "E_REPLY", "message": "scale must be a positive integer"}
            if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
                return {"ok": False, "code": "E_REPLY", "message": "group_modulus must be a positive integer"}
            matrix = np.asarray(samples, dtype=np.int64)
            if matrix.size == 0:
                matrix = matrix.reshape(0, 2)
            if matrix.ndim != 2 or matrix.shape[1] != 2:
                return {"ok": False, "code": "E_REPLY", "message": "samples must be [key,value] pairs"}
            keys = matrix[:, 0]
            values = matrix[:, 1]
            summary_groups: list[dict[str, int]] = []
            for key in sorted(int(value) for value in np.unique(keys).tolist()):
                selected = values[keys == np.int64(key)]
                count = int(selected.size)
                total = int(selected.sum(dtype=np.int64))
                summary_groups.append(
                    {
                        "key": key,
                        "count": count,
                        "sum": total,
                        "min": int(selected.min()),
                        "max": int(selected.max()),
                        "mean_fixed": (total * int(scale)) // count,
                    }
                )
            summary = {
                "groups": summary_groups,
                "sample_count": int(matrix.shape[0]),
                "scale": int(scale),
                "group_modulus": int(groups),
            }
            summary["digest"] = digest_obj(summary)
            elapsed = max(0, (time.perf_counter_ns() - started) // 1_000_000)
            with self.lock:
                current = self.leases.get(str(lease_id))
                if current is None or current.state != "active":
                    return {"ok": False, "code": "E_LEASE", "message": "lease expired or was released during compute"}
                current.state = "completed"
            return {
                "ok": True,
                "node_id": self.node_id,
                "shard_id": request["shard_id"],
                "config_version": request["config_version"],
                "lease_id": lease_id,
                "lease_version": request["lease_version"],
                "summary": summary,
                "elapsed_monotonic_ms": elapsed,
            }
        finally:
            with self.lock:
                self.active_computes = max(0, self.active_computes - 1)


STATE: NodeState | None = None


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline(2_000_000)
            if not raw:
                return
            request: Any = None
            try:
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                reply = dispatch_request(request)
            except Exception as exc:  # malformed protocol input must not kill the node
                reply = {
                    "ok": False,
                    "code": "E_REPLY",
                    "message": f"invalid request: {exc}",
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                }
            if isinstance(request, dict) and "request_id" in request and "request_id" not in reply:
                reply["request_id"] = request["request_id"]
            try:
                self.wfile.write(_line(reply))
                self.wfile.flush()
            except OSError:
                return
            if reply.get("_shutdown"):
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return


def dispatch_request(request: dict[str, Any]) -> dict[str, Any]:
    assert STATE is not None
    op = request.get("op")
    if op == "ping":
        return STATE.ping()
    if op == "snapshot":
        return {"ok": True, **STATE.snapshot()}
    if op == "configure":
        return STATE.configure(request)
    if op == "lease_acquire":
        return STATE.acquire_lease(request)
    if op == "lease_renew":
        return STATE.renew_lease(request)
    if op == "lease_release":
        return STATE.release_lease(request)
    if op == "compute":
        return STATE.compute(request)
    if op == "shutdown":
        return {"ok": True, "node_id": STATE.node_id, "_shutdown": True}
    return {"ok": False, "code": "E_REPLY", "message": f"unknown op: {op}"}


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _watch_parent(parent_pid: int) -> None:
    while True:
        time.sleep(0.1)
        if parent_pid <= 0:
            return
        if os.name == "nt":
            import ctypes

            synchronize = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, parent_pid)
            if not handle:
                os._exit(0)
            result = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
            ctypes.windll.kernel32.CloseHandle(handle)
            if result == 0:  # WAIT_OBJECT_0
                os._exit(0)
        else:
            try:
                os.kill(parent_pid, 0)
            except OSError:
                os._exit(0)


def main(argv: list[str] | None = None) -> int:
    global STATE
    parser = argparse.ArgumentParser(description="quota ledger node worker")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--lease-limit", required=True, type=int)
    parser.add_argument("--capacity-version", required=True, type=int)
    parser.add_argument("--default-ttl-ms", required=True, type=int)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args(argv)
    STATE = NodeState(
        node_id=args.node_id,
        seed=args.seed,
        concurrency=args.concurrency,
        lease_limit=args.lease_limit,
        capacity_version=args.capacity_version,
        default_ttl_ms=args.default_ttl_ms,
    )
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        sys.stderr.write(
            canonical_json({"event": "error", "code": "E_PORT", "node_id": args.node_id, "message": str(exc)}) + "\n"
        )
        sys.stderr.flush()
        return 3
    if args.parent_pid:
        threading.Thread(target=_watch_parent, args=(args.parent_pid,), daemon=True).start()
    sys.stdout.write(
        canonical_json(
            {
                "event": "ready",
                "node_id": args.node_id,
                "port": args.port,
                "pid": os.getpid(),
            }
        )
        + "\n"
    )
    sys.stdout.flush()
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

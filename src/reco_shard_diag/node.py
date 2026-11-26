from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import BizError, make_error
from .fixedpoint import aggregate_groups
from .models import NodeSpec


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class LeaseRecord:
    lease_id: str
    shard_id: str
    generation: int
    expires_at_ms: int
    issued_at_ms: int
    renewed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "generation": self.generation,
            "expires_at_ms": self.expires_at_ms,
            "issued_at_ms": self.issued_at_ms,
            "renewed_count": self.renewed_count,
        }


@dataclass
class NodeRuntime:
    spec: NodeSpec
    clock_ms: Callable[[], int] = monotonic_ms
    lease_generation: int = 0
    leases: dict[str, LeaseRecord] = field(default_factory=dict)
    expired_leases: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_by_shard: dict[str, str] = field(default_factory=dict)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    errors: list[BizError] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    ids: int = 0

    def _record(self, event_type: str, **payload: Any) -> None:
        self.ledger.append(
            {"sequence": len(self.ledger) + 1, "event": event_type, **payload}
        )

    def _error(self, error: BizError) -> dict[str, Any]:
        self.errors.append(error)
        self._record("error", error=error.to_dict())
        return {
            "ok": False,
            "node_id": self.spec.node_id,
            "lease_generation": self.lease_generation,
            **error.to_dict(),
        }

    def _next_id(self, prefix: str) -> str:
        self.ids += 1
        return f"{prefix}-{self.spec.node_id}-{self.ids:06d}"

    def _expire_locked(self, now_ms: int) -> list[dict[str, Any]]:
        expired: list[dict[str, Any]] = []
        for lease_id, lease in sorted(self.leases.items()):
            if now_ms >= lease.expires_at_ms:
                self.leases.pop(lease_id, None)
                error = make_error(
                    "E_LEASE_EXPIRED",
                    "lease",
                    lease_id,
                    "expires_at_ms",
                    "租约超过到期毫秒，由回收逻辑释放工作单元",
                    details={
                        "node_id": self.spec.node_id,
                        "shard_id": lease.shard_id,
                        "generation": lease.generation,
                        "expires_at_ms": lease.expires_at_ms,
                        "swept_at_ms": now_ms,
                    },
                )
                self.errors.append(error)
                record = {
                    "lease_id": lease_id,
                    "shard_id": lease.shard_id,
                    "generation": lease.generation,
                    "expires_at_ms": lease.expires_at_ms,
                    "swept_at_ms": now_ms,
                    "code": error.code,
                }
                self.expired_leases[lease_id] = record
                self._record("lease_expired", **record)
                expired.append(record)
        return expired

    def sweep(self) -> dict[str, Any]:
        with self.lock:
            now_ms = int(self.clock_ms())
            expired = self._expire_locked(now_ms)
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_generation": self.lease_generation,
                "expired": expired,
                "active_work_units": len(self.leases),
                "now_ms": now_ms,
            }

    def status(self) -> dict[str, Any]:
        with self.lock:
            now_ms = int(self.clock_ms())
            expired = self._expire_locked(now_ms)
            capacity = min(self.spec.capacity_slots, self.spec.lease_limit)
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_generation": self.lease_generation,
                "capacity_slots": self.spec.capacity_slots,
                "lease_limit": self.spec.lease_limit,
                "effective_active_limit": capacity,
                "active_work_units": len(self.leases),
                "available_work_units": max(0, capacity - len(self.leases)),
                "leases": [
                    lease.to_dict()
                    for lease in sorted(self.leases.values(), key=lambda item: item.lease_id)
                ],
                "expired": expired,
                "errors_count": len(self.errors),
                "now_ms": now_ms,
            }

    def ledger_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_generation": self.lease_generation,
                "active_work_units": len(self.leases),
                "leases": [
                    lease.to_dict()
                    for lease in sorted(self.leases.values(), key=lambda item: item.lease_id)
                ],
                "latest_by_shard": dict(sorted(self.latest_by_shard.items())),
                "ledger": list(self.ledger),
                "errors": [error.to_dict() for error in self.errors],
            }

    def acquire(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now_ms = int(self.clock_ms())
            self._expire_locked(now_ms)
            expected = request.get("expected_generation")
            if expected is not None and expected != self.lease_generation:
                return self._error(
                    make_error(
                        "E_LEASE_GENERATION_MISMATCH",
                        "node",
                        self.spec.node_id,
                        "expected_generation",
                        f"期望代次 {expected}，当前公认代次 {self.lease_generation}",
                    )
                )
            capacity = min(self.spec.capacity_slots, self.spec.lease_limit)
            if len(self.leases) >= capacity:
                code = (
                    "E_NODE_LEASE_LIMIT"
                    if self.spec.lease_limit <= self.spec.capacity_slots
                    else "E_NODE_CAPACITY_EXHAUSTED"
                )
                return self._error(
                    make_error(
                        code,
                        "node",
                        self.spec.node_id,
                        "lease_limit/capacity_slots",
                        "在途工作单元已达节点上限",
                        details={
                            "active_work_units": len(self.leases),
                            "capacity_slots": self.spec.capacity_slots,
                            "lease_limit": self.spec.lease_limit,
                        },
                    )
                )
            ttl_ms = request.get("ttl_ms")
            if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
                return self._error(
                    make_error(
                        "E_PROTOCOL", "request", "acquire", "ttl_ms", "ttl_ms 必须为正整数"
                    )
                )
            shard_id = request.get("shard_id")
            if not isinstance(shard_id, str) or not shard_id:
                return self._error(
                    make_error(
                        "E_PROTOCOL", "request", "acquire", "shard_id", "shard_id 必须为非空字符串"
                    )
                )
            self.lease_generation += 1
            lease_id = self._next_id("lease")
            lease = LeaseRecord(
                lease_id=lease_id,
                shard_id=shard_id,
                generation=self.lease_generation,
                expires_at_ms=now_ms + ttl_ms,
                issued_at_ms=now_ms,
            )
            self.leases[lease_id] = lease
            self._record(
                "lease_acquired", lease=lease.to_dict(), node_generation=self.lease_generation
            )
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_id": lease_id,
                "lease_generation": self.lease_generation,
                "expires_at_ms": lease.expires_at_ms,
                "active_work_units": len(self.leases),
            }

    def renew(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now_ms = int(self.clock_ms())
            self._expire_locked(now_ms)
            lease_id = str(request.get("lease_id", ""))
            expected = request.get("expected_generation")
            lease = self.leases.get(lease_id)
            if lease is None:
                if lease_id in self.expired_leases:
                    return self._error(
                        make_error(
                            "E_LEASE_EXPIRED",
                            "lease",
                            lease_id,
                            "expires_at_ms",
                            "续期前租约已过期并由回收逻辑释放",
                            details={"swept_at_ms": now_ms},
                        )
                    )
                return self._error(
                    make_error(
                        "E_LEASE_NOT_FOUND",
                        "lease",
                        lease_id,
                        "lease_id",
                        "租约不存在或已释放",
                    )
                )
            if expected != lease.generation:
                return self._error(
                    make_error(
                        "E_LEASE_GENERATION_MISMATCH",
                        "lease",
                        lease_id,
                        "expected_generation",
                        f"期望代次 {expected}，当前公认代次 {self.lease_generation}",
                    )
                )
            if now_ms >= lease.expires_at_ms:
                return self._error(
                    make_error(
                        "E_LEASE_EXPIRED",
                        "lease",
                        lease_id,
                        "expires_at_ms",
                        "租约已超过到期毫秒",
                    )
                )
            ttl_ms = request.get("ttl_ms")
            if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
                return self._error(
                    make_error(
                        "E_PROTOCOL", "request", "renew", "ttl_ms", "ttl_ms 必须为正整数"
                    )
                )
            lease.expires_at_ms = now_ms + ttl_ms
            lease.renewed_count += 1
            self._record("lease_renewed", lease=lease.to_dict())
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_id": lease_id,
                "lease_generation": self.lease_generation,
                "expires_at_ms": lease.expires_at_ms,
                "renewed_count": lease.renewed_count,
            }

    def release(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now_ms = int(self.clock_ms())
            self._expire_locked(now_ms)
            lease_id = str(request.get("lease_id", ""))
            expected = request.get("expected_generation")
            lease = self.leases.get(lease_id)
            if lease is None:
                if lease_id in self.expired_leases:
                    return self._error(
                        make_error(
                            "E_LEASE_EXPIRED",
                            "lease",
                            lease_id,
                            "expires_at_ms",
                            "释放前租约已过期并由回收逻辑释放",
                        )
                    )
                return self._error(
                    make_error(
                        "E_LEASE_NOT_FOUND",
                        "lease",
                        lease_id,
                        "lease_id",
                        "租约不存在或已释放",
                    )
                )
            if expected != lease.generation:
                return self._error(
                    make_error(
                        "E_LEASE_GENERATION_MISMATCH",
                        "lease",
                        lease_id,
                        "expected_generation",
                        f"期望代次 {expected}，当前公认代次 {self.lease_generation}",
                    )
                )
            self.leases.pop(lease_id, None)
            self._record(
                "lease_released",
                lease_id=lease_id,
                shard_id=lease.shard_id,
                generation=lease.generation,
                released_at_ms=now_ms,
            )
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_id": lease_id,
                "lease_generation": self.lease_generation,
                "active_work_units": len(self.leases),
            }

    def aggregate(self, request: dict[str, Any]) -> dict[str, Any]:
        started = int(self.clock_ms())
        with self.lock:
            now_ms = started
            self._expire_locked(now_ms)
            lease_id = str(request.get("lease_id", ""))
            expected = request.get("expected_generation")
            lease = self.leases.get(lease_id)
            if lease is None:
                code = (
                    "E_LEASE_EXPIRED"
                    if lease_id in self.expired_leases
                    else "E_LEASE_NOT_FOUND"
                )
                return self._error(
                    make_error(
                        code,
                        "lease",
                        lease_id,
                        "lease_id",
                        "聚合请求没有可用租约",
                    )
                )
            if expected != lease.generation:
                return self._error(
                    make_error(
                        "E_LEASE_GENERATION_MISMATCH",
                        "lease",
                        lease_id,
                        "expected_generation",
                        f"期望代次 {expected}，当前公认代次 {self.lease_generation}",
                    )
                )
            shard_id = request.get("shard_id")
            request_id = request.get("request_id")
            dimensions = request.get("dimensions")
            width = request.get("width")
            fixed_scale = request.get("fixed_scale", 1000)
            samples = request.get("samples")
            if not isinstance(shard_id, str) or not isinstance(request_id, str):
                return self._error(
                    make_error(
                        "E_PROTOCOL",
                        "request",
                        "aggregate",
                        "shard_id/request_id",
                        "标识字段不合法",
                    )
                )
            if (
                isinstance(dimensions, bool)
                or not isinstance(dimensions, int)
                or dimensions <= 0
                or isinstance(width, bool)
                or not isinstance(width, int)
                or not 2 <= width <= 62
                or isinstance(fixed_scale, bool)
                or not isinstance(fixed_scale, int)
                or fixed_scale <= 0
                or not isinstance(samples, list)
            ):
                return self._error(
                    make_error(
                        "E_PROTOCOL",
                        "request",
                        "aggregate",
                        "width/dimensions/samples",
                        "固定点或样本字段不合法",
                    )
                )
            for sample in samples:
                values = sample.get("values") if isinstance(sample, dict) else None
                if not isinstance(values, list) or len(values) != dimensions:
                    return self._error(
                        make_error(
                            "E_SAMPLE_DIMENSION",
                            "shard",
                            shard_id,
                            "samples.values",
                            f"样本维度与声明 {dimensions} 不一致",
                        )
                    )
            try:
                summary = aggregate_groups(
                    samples=samples,
                    dimensions=dimensions,
                    width=width,
                    fixed_scale=fixed_scale,
                )
            except (TypeError, ValueError) as exc:
                return self._error(
                    make_error("E_SAMPLE_DIMENSION", "shard", shard_id, "samples", str(exc))
                )
            previous_request = self.latest_by_shard.get(shard_id)
            duplicate = previous_request is not None
            self.latest_by_shard[shard_id] = request_id
            elapsed = max(0, int(self.clock_ms()) - started)
            response = {
                "ok": True,
                "node_id": self.spec.node_id,
                "request_id": request_id,
                "shard_id": shard_id,
                "batch_id": request.get("batch_id"),
                "quota_version": request.get("quota_version"),
                "width": width,
                "dimensions": dimensions,
                "lease_id": lease_id,
                "lease_generation": self.lease_generation,
                "elapsed_monotonic_ms": elapsed,
                "aggregate": summary,
                "duplicate_delivery": duplicate,
                "supersedes_request_id": previous_request,
            }
            self._record(
                "aggregate_result",
                request_id=request_id,
                shard_id=shard_id,
                lease_generation=self.lease_generation,
                duplicate=duplicate,
                supersedes_request_id=previous_request,
                elapsed_monotonic_ms=elapsed,
            )
            return response

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "ping":
            return {
                "ok": True,
                "node_id": self.spec.node_id,
                "lease_generation": self.lease_generation,
            }
        if operation == "status":
            return self.status()
        if operation == "ledger":
            return self.ledger_snapshot()
        if operation == "sweep":
            return self.sweep()
        if operation == "acquire":
            return self.acquire(request)
        if operation == "renew":
            return self.renew(request)
        if operation == "release":
            return self.release(request)
        if operation == "aggregate":
            return self.aggregate(request)
        return {
            "ok": False,
            "node_id": self.spec.node_id,
            "lease_generation": self.lease_generation,
            **make_error(
                "E_PROTOCOL", "request", str(operation), "operation", "未知协议操作"
            ).to_dict(),
        }


class _JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        runtime: NodeRuntime = self.server.runtime  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be object")
                response = runtime.dispatch(request)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                response = {
                    "ok": False,
                    "node_id": runtime.spec.node_id,
                    "lease_generation": runtime.lease_generation,
                    **make_error(
                        "E_PROTOCOL", "request", "line", "json", str(exc)
                    ).to_dict(),
                }
            self.wfile.write(
                json.dumps(response, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            )
            self.wfile.flush()


class NodeTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: NodeRuntime):
        self.runtime = runtime
        super().__init__(address, _JsonLineHandler)


def _set_parent_death_signal() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)
    except Exception:
        return


def run_server(spec: NodeSpec, *, lease_generation: int = 0) -> None:
    _set_parent_death_signal()
    runtime = NodeRuntime(spec=spec, lease_generation=lease_generation)
    server = NodeTcpServer((spec.host, spec.port), runtime)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True, name="node-server")
    server_thread.start()
    sys.stdout.write(
        json.dumps(
            {
                "event": "READY",
                "node_id": spec.node_id,
                "host": spec.host,
                "port": spec.port,
                "pid": os.getpid(),
                "lease_generation": lease_generation,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    sys.stdout.flush()
    try:
        while server_thread.is_alive():
            server_thread.join(0.2)
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--lease-generation", type=int, default=0)
    args = parser.parse_args(argv)
    with open(args.spec_file, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    spec = NodeSpec(
        node_id=raw["node_id"],
        host=raw["host"],
        port=int(raw["port"]),
        capacity_slots=int(raw["capacity_slots"]),
        lease_limit=int(raw["lease_limit"]),
    )
    run_server(spec, lease_generation=args.lease_generation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

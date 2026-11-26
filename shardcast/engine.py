"""Parallel scheduling engine with leases, budgets and deterministic ordering."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .capacity import CapacityManager, mark_stale
from .clock import now_ms
from .errors import LeaseError, PortError, TimeoutError, FieldError
from .ledger import Ledger
from .leases import Lease, LeaseManager
from .protocol import request_dispatch, request_renew
from .sample import generate_sample
from .spec import Spec, topological_order
from .worker_pool import WorkerPool

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class ShardRuntime:
    shard_id: str
    status: str = "waiting"
    node_id: str | None = None
    lease_id: str | None = None
    lease_version: int | None = None
    capacity_version: int | None = None
    summary: dict | None = None
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    attempts: int = 0
    frozen: bool = False
    stale: bool = False
    fail_reason: str | None = None


@dataclass
class EngineResult:
    outcomes: dict[str, dict]
    decisions: list[dict]
    capacity_diffs: list[dict]
    waits: dict[str, dict]
    water_marks: dict[str, dict]
    capacity_version: int


class Engine:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        *,
        socket_timeout: float = 3.0,
        max_attempts: int = 3,
        batch_size: int = 64,
    ):
        self.spec = spec
        self.ledger = ledger
        self.socket_timeout = socket_timeout
        self.max_attempts = max_attempts
        self.batch_size = batch_size
        self.pool = WorkerPool(spec)
        self.lease_manager = LeaseManager(spec.lease_ttl_ms)
        self.capacity = CapacityManager(spec)
        self.runtimes: dict[str, ShardRuntime] = {
            shard.shard_id: ShardRuntime(shard.shard_id) for shard in spec.shards
        }
        self.arrived: set[str] = set()
        self.cancelled: set[str] = set()
        self.frozen: set[str] = set()
        self.in_flight: dict[str, Lease] = {}
        self._lock = threading.Lock()
        self._lease_seq = 0
        self._resource_used = 0
        self.water_marks: dict[str, dict] = {}
        self.waits: dict[str, dict] = {}
        self._event_index = -1

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        created, replacement = self.ledger.initialize_snapshot(
            spec_digest=self.spec.structural_digest(),
            node_set_version=self.spec.node_set_version(),
            capacity_version=self.capacity.current.version,
        )
        self.snapshot_replacement = replacement
        self.pool.start_all()

    def stop(self) -> None:
        self.pool.stop_all()

    # ---- determinism helpers ------------------------------------------
    def _next_lease_id(self) -> str:
        self._lease_seq += 1
        return f"lease-{self._lease_seq:06d}"

    def _candidate_node(self, shard_id: str) -> str:
        index = self.spec.shard_ids.index(shard_id)
        node = self.spec.nodes[index % len(self.spec.nodes)]
        return node.node_id

    def _topo_index(self, shard_id: str) -> int:
        return topological_order(self.spec).index(shard_id)

    # ---- readiness and constraints ------------------------------------
    def _dependencies_terminal(self, shard_id: str) -> bool:
        shard = self.spec.shard(shard_id)
        for upstream in shard.depends_on:
            runtime = self.runtimes[upstream]
            if runtime.status not in TERMINAL_STATUSES or runtime.summary is None:
                return False
        return True

    def _can_assign(self, shard_id: str, node_id: str) -> tuple[bool, str]:
        shard = self.spec.shard(shard_id)
        view = self.capacity.current
        active_leases = self.lease_manager.active_for_node(node_id)
        quota = view.quota(node_id)
        if len(active_leases) >= quota.lease_limit:
            return False, "node_lease_limit"
        active_dispatched = [
            lease for lease in active_leases if lease.shard_id in self.in_flight
        ]
        if len(active_dispatched) >= quota.concurrency:
            return False, "node_concurrency"
        if self._resource_used + shard.resource_units > view.global_budget:
            return False, "global_budget"
        return True, "ok"

    def _ready_shards(self) -> list[str]:
        ready: list[str] = []
        for shard_id in self.arrived:
            runtime = self.runtimes[shard_id]
            if runtime.status != "waiting":
                continue
            if shard_id in self.frozen or shard_id in self.cancelled:
                continue
            if self.lease_manager.owner_of(shard_id) is not None:
                continue
            if not self._dependencies_terminal(shard_id):
                continue
            ready.append(shard_id)
        ready.sort(
            key=lambda shard_id: (
                self.spec.shard(shard_id).priority,
                self._topo_index(shard_id),
                shard_id,
            )
        )
        return ready

    # ---- dispatch ------------------------------------------------------
    def _build_batches(self, node_id: str, shard_id: str) -> list[list[list[int]]]:
        node = self.spec.node(node_id)
        shard = self.spec.shard(shard_id)
        data = generate_sample(node.seed, shard_id, shard.sample_rows)
        batches: list[list[list[int]]] = []
        for start in range(0, data.shape[0], self.batch_size):
            part = data[start:start + self.batch_size]
            batches.append(part.tolist())
        return batches

    def _dispatch_one(self, shard_id: str, node_id: str, lease: Lease) -> dict:
        node = self.spec.node(node_id)
        request = {
            "type": "dispatch",
            "lease_id": lease.lease_id,
            "lease_version": lease.lease_version,
            "node_id": node_id,
            "shard_id": shard_id,
            "capacity_version": lease.capacity_version,
            "sample_rows": self.spec.shard(shard_id).sample_rows,
            "batches": self._build_batches(node_id, shard_id),
        }
        return request_dispatch(node.host, node.port, request, self.socket_timeout)

    def _schedule_pass(self) -> list[str]:
        """One single-dispatch pass: a shard is launched at most once here."""
        launched: list[str] = []
        ready = self._ready_shards()
        for shard_id in ready:
            if shard_id in self.in_flight:
                continue
            node_id = self._candidate_node(shard_id)
            allowed, reason = self._can_assign(shard_id, node_id)
            runtime = self.runtimes[shard_id]
            if not allowed:
                if runtime.wait_start_ms is None:
                    runtime.wait_start_ms = now_ms()
                    self.waits[shard_id] = {
                        "shard_id": shard_id,
                        "wait_start_ms": runtime.wait_start_ms,
                        "wait_end_ms": None,
                        "reason": reason,
                    }
                continue
            lease = self.lease_manager.grant(
                self._next_lease_id(),
                shard_id,
                node_id,
                self.capacity.current.version,
            )
            self.pool.install_lease(node_id, lease.to_dict())
            with self._lock:
                self.in_flight[shard_id] = lease
                self._resource_used += self.spec.shard(shard_id).resource_units
            runtime.status = "dispatched"
            runtime.node_id = node_id
            runtime.lease_id = lease.lease_id
            runtime.lease_version = lease.lease_version
            runtime.capacity_version = lease.capacity_version
            runtime.attempts += 1
            if runtime.wait_start_ms is not None:
                runtime.wait_end_ms = now_ms()
                self.waits[shard_id]["wait_end_ms"] = runtime.wait_end_ms
            launched.append(shard_id)
        return launched

    def _run_dispatched(self, launched: list[str]) -> None:
        if not launched:
            return
        workers = min(max(1, len(launched)), 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._dispatch_one, shard_id, self.runtimes[shard_id].node_id,
                                self.in_flight[shard_id]): shard_id
                for shard_id in launched
            }
            for future in futures:
                shard_id = futures[future]
                try:
                    receipt = future.result()
                except (PortError, TimeoutError, FieldError) as exc:
                    self._handle_dispatch_failure(shard_id, exc)
                    continue
                self._handle_receipt(shard_id, receipt)

    def _handle_receipt(self, shard_id: str, receipt: dict) -> None:
        runtime = self.runtimes[shard_id]
        lease = self.in_flight.pop(shard_id, None)
        if not receipt.get("accepted"):
            self.lease_manager.release(receipt["lease_id"], reason=receipt.get("error", "rejected"))
            with self._lock:
                self._resource_used -= self.spec.shard(shard_id).resource_units
            self._requeue(shard_id, receipt.get("error", "node_rejected"))
            return
        summary = receipt["summary"]
        with self._lock:
            self._resource_used -= self.spec.shard(shard_id).resource_units
        runtime.status = "completed"
        runtime.summary = summary
        runtime.capacity_version = receipt["capacity_version"]
        runtime.stale = receipt["capacity_version"] < self.capacity.current.version
        if lease is not None:
            self.lease_manager.release(lease.lease_id, reason="completed")
        self._record_water_mark(runtime, receipt.get("elapsed_ms", 0))
        self._commit_runtime(runtime)

    def _handle_dispatch_failure(self, shard_id: str, exc: Exception) -> None:
        lease = self.in_flight.pop(shard_id, None)
        with self._lock:
            self._resource_used -= self.spec.shard(shard_id).resource_units
        if lease is not None:
            self.lease_manager.release(lease.lease_id, reason="transport_error")
        self._requeue(shard_id, str(exc))

    def _requeue(self, shard_id: str, reason: str) -> None:
        runtime = self.runtimes[shard_id]
        if shard_id in self.cancelled:
            runtime.status = "cancelled"
            return
        if runtime.attempts >= self.max_attempts:
            runtime.status = "failed"
            runtime.fail_reason = reason
            self._commit_runtime(runtime)
            return
        runtime.status = "waiting"
        runtime.node_id = None
        runtime.lease_id = None
        runtime.lease_version = None

    def _record_water_mark(self, runtime: ShardRuntime, elapsed_ms: int) -> None:
        node_id = runtime.node_id
        version = runtime.capacity_version
        mark = self.water_marks.get(node_id)
        if mark is None or mark["capacity_version"] != version:
            mark = {
                "node_id": node_id,
                "capacity_version": version,
                "high_water_ms": 0,
                "shard_count": 0,
                "resource_total": 0,
            }
        mark["high_water_ms"] = max(int(mark["high_water_ms"]), int(now_ms()))
        mark["shard_count"] += 1
        mark["resource_total"] += self.spec.shard(runtime.shard_id).resource_units
        self.water_marks[node_id] = mark

    def _commit_runtime(self, runtime: ShardRuntime) -> None:
        outcome = {
            "shard_id": runtime.shard_id,
            "node_id": runtime.node_id or "",
            "status": runtime.status,
            "capacity_version": runtime.capacity_version or self.capacity.current.version,
            "lease_id": runtime.lease_id,
            "lease_version": None,
            "summary": runtime.summary,
            "resource_units": self.spec.shard(runtime.shard_id).resource_units,
            "committed_ms": now_ms(),
        }
        outcome["lease_version"] = runtime.lease_version
        with self._lock:
            self.ledger.commit_outcome(
                outcome=outcome,
                water_marks=self.water_marks,
                last_seq=self._event_index,
                capacity_version=self.capacity.current.version,
            )

    # ---- events --------------------------------------------------------
    def _apply_event(self, event) -> None:
        self._event_index = event.seq
        if event.kind == "arrive":
            self.arrived.add(event.shard_id)
            runtime = self.runtimes[event.shard_id]
            if runtime.status == "waiting" and runtime.wait_start_ms is None:
                runtime.wait_start_ms = now_ms()
                self.waits[event.shard_id] = {
                    "shard_id": event.shard_id,
                    "wait_start_ms": runtime.wait_start_ms,
                    "wait_end_ms": None,
                    "reason": "queued",
                }
        elif event.kind == "freeze":
            self.frozen.add(event.shard_id)
            self.runtimes[event.shard_id].frozen = True
        elif event.kind == "unfreeze":
            self.frozen.discard(event.shard_id)
            self.runtimes[event.shard_id].frozen = False
        elif event.kind == "cancel":
            self._cancel_shard(event.shard_id)
        elif event.kind == "renew":
            self._renew_lease(event.shard_id, event.lease_id)
        elif event.kind == "capacity":
            self._replace_capacity(event.capacity_version)
        elif event.kind == "_sleep":
            time.sleep(int(getattr(event, "ms", 20)) / 1000.0)

    def _cancel_shard(self, shard_id: str) -> None:
        self.cancelled.add(shard_id)
        runtime = self.runtimes[shard_id]
        owner = self.lease_manager.owner_of(shard_id)
        if owner is not None and shard_id in self.in_flight:
            lease = self.in_flight.pop(shard_id)
            with self._lock:
                self._resource_used -= self.spec.shard(shard_id).resource_units
            self.lease_manager.release(lease.lease_id, reason="cancelled")
        runtime.status = "cancelled"
        runtime.fail_reason = "cancelled"
        self._commit_runtime(runtime)

    def _renew_lease(self, shard_id: str, lease_id: str) -> None:
        lease = self.lease_manager.get(lease_id)
        if lease is None or lease.shard_id != shard_id:
            self.lease_manager.reject_renewal(lease_id, "unknown_lease")
            raise LeaseError(f"renew failed: unknown lease {lease_id}", path=f"leases.{lease_id}")
        node = self.spec.node(lease.node_id)
        request = {
            "type": "renew",
            "lease_id": lease_id,
            "lease_version": lease.lease_version,
            "node_id": lease.node_id,
            "shard_id": shard_id,
            "capacity_version": lease.capacity_version,
        }
        receipt = request_renew(node.host, node.port, request, self.socket_timeout)
        if not receipt.get("accepted"):
            self.lease_manager.reject_renewal(lease_id, receipt.get("error", "version_mismatch"))
            raise LeaseError(
                f"renew rejected for {lease_id}",
                path=f"leases.{lease_id}.lease_version",
            )
        self.lease_manager.apply_renewal_receipt(lease_id, receipt)

    def _replace_capacity(self, new_version: int) -> None:
        if new_version == self.capacity.current.version:
            return
        previous, new_view, diff = self.capacity.replace(new_version)
        revoked = self.lease_manager.revoke_before_version(new_version)
        for lease in revoked:
            self.in_flight.pop(lease.shard_id, None)
            with self._lock:
                units = self.spec.shard(lease.shard_id).resource_units
                self._resource_used = max(0, self._resource_used - units)
            runtime = self.runtimes[lease.shard_id]
            if runtime.status == "dispatched":
                runtime.status = "waiting"
                runtime.node_id = None
                runtime.lease_id = None
                runtime.lease_version = None
                runtime.attempts -= 0
        for runtime in self.runtimes.values():
            if runtime.capacity_version is not None and runtime.capacity_version < new_version:
                runtime.stale = True
        for node_id, mark in list(self.water_marks.items()):
            if mark["capacity_version"] < new_version:
                self.water_marks[node_id] = mark_stale(mark, new_version)

    # ---- main loop -----------------------------------------------------
    def run(self) -> EngineResult:
        try:
            self.start()
            for event in self.spec.events:
                self._apply_event(event)
                self._drain_once()
            self._drain_until_settled()
            return self._build_result()
        finally:
            self.stop()

    def _drain_once(self) -> None:
        self.lease_manager.expire_due()
        self._reclaim_expired_inflight()
        launched = self._schedule_pass()
        self._run_dispatched(launched)

    def _reclaim_expired_inflight(self) -> None:
        for shard_id, lease in list(self.in_flight.items()):
            if self.lease_manager.is_expired(lease.lease_id):
                self.in_flight.pop(shard_id, None)
                with self._lock:
                    self._resource_used -= self.spec.shard(shard_id).resource_units
                self._requeue(shard_id, "lease_expired")

    def _drain_until_settled(self) -> None:
        for _ in range(1000):
            self._drain_once()
            pending = [
                shard_id
                for shard_id in self.arrived
                if self.runtimes[shard_id].status not in TERMINAL_STATUSES
                and shard_id not in self.frozen
            ]
            if not pending and not self.in_flight:
                break

    def _build_result(self) -> EngineResult:
        outcomes: dict[str, dict] = {}
        for shard_id, runtime in self.runtimes.items():
            row = self.ledger.get_outcome(shard_id)
            outcomes[shard_id] = {
                "shard_id": shard_id,
                "status": runtime.status,
                "node_id": runtime.node_id,
                "capacity_version": runtime.capacity_version,
                "stale": runtime.stale,
                "summary_digest": None
                if runtime.summary is None or row is None
                else row["summary_digest"],
            }
        return EngineResult(
            outcomes=outcomes,
            decisions=self.lease_manager.decisions_snapshot(),
            capacity_diffs=self.capacity.diffs,
            waits=self.waits,
            water_marks=self.water_marks,
            capacity_version=self.capacity.current.version,
        )

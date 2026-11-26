"""Parallel probe scheduling with real leases and monotonic deadlines."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .canonical import first_field_difference, sha256_digest
from .errors import E_LEASE, E_REPLY, RouteError
from .probe import expected_probe_receipt
from .spec import RoutingSpec
from .worker import WorkerClient


ACTIVE = "active"
COMPLETED = "completed"
EXPIRED = "expired"
VERSION_INVALIDATED = "version_invalidated"
FAILED = "failed"
FINAL_STATUSES = {COMPLETED, EXPIRED, VERSION_INVALIDATED, FAILED}


@dataclass
class Lease:
    lease_id: str
    database_id: str
    event_seq: int
    route_version: int
    batch_id: str
    created_ns: int = 0
    deadline_ns: int = 0
    lease_version: int = 0
    status: str = ACTIVE
    decision_reason: str | None = None
    decision_event_seq: int | None = None
    decision_route_version: int | None = None
    wait_started_ms: int = 0
    wait_ended_ms: int = 0
    waited: bool = False
    inflight_released: bool = False

    def ownership(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "database_id": self.database_id,
            "event_seq": self.event_seq,
            "route_version": self.route_version,
            "lease_version": self.lease_version,
            "status": self.status,
            "decision_reason": self.decision_reason,
            "decision_event_seq": self.decision_event_seq,
            "decision_route_version": self.decision_route_version,
        }

    def runtime(self) -> dict[str, Any]:
        return {
            "created_monotonic_ms": self.created_ns // 1_000_000,
            "deadline_monotonic_ms": self.deadline_ns // 1_000_000,
            "queue_wait_started_monotonic_ms": self.wait_started_ms,
            "queue_wait_ended_monotonic_ms": self.wait_ended_ms,
            "queued": self.waited,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.ownership(), "runtime": self.runtime()}


@dataclass
class ProbeOutcome:
    event_seq: int
    database_id: str
    keys: list[dict[str, Any]]
    receipt: dict[str, Any]
    lease: Lease

    def deterministic(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "database_id": self.database_id,
            "keys": self.keys,
            "receipt": {
                key: self.receipt[key]
                for key in ("batch_id", "database_id", "route_version", "probe_digest", "fixedpoint_checksum_u64", "hits")
            },
            "lease": self.lease.ownership(),
        }


class LeaseScheduler:
    """Triple-constrained scheduler.

    A slot is granted only when all three predicates hold at once:
    global active leases, per-database active leases, and per-database in-flight
    probes.  Counters are decremented exactly once through the lease state
    machine.
    """

    def __init__(self, spec: RoutingSpec, workers: dict[str, WorkerClient]) -> None:
        self.spec = spec
        self.workers = workers
        self._condition = asyncio.Condition()
        self._leases: dict[str, Lease] = {}
        self._history: list[Lease] = []
        self._active_global = 0
        self._active_db: dict[str, int] = {database.id: 0 for database in spec.databases}
        self._inflight_db: dict[str, int] = {database.id: 0 for database in spec.databases}
        self._peak_inflight_db: dict[str, int] = {database.id: 0 for database in spec.databases}
        self._monitor_task: asyncio.Task[Any] | None = None
        self._closed = False
        self._route_version = spec.route_version

    async def __aenter__(self) -> "LeaseScheduler":
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            now_ns = time.monotonic_ns()
            for lease in self._leases.values():
                if lease.status == ACTIVE:
                    lease.status = FAILED
                    lease.decision_reason = "scheduler_closed"
                    lease.decision_event_seq = lease.event_seq
                    lease.decision_route_version = lease.route_version
                    self._release_active_locked(lease)
            self._condition.notify_all()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    def _release_active_locked(self, lease: Lease) -> None:
        if lease.status == ACTIVE:
            lease.status = FAILED
            lease.decision_reason = "internal_state_machine_error"
        if self._active_db.get(lease.database_id, 0) > 0:
            self._active_db[lease.database_id] -= 1
        if self._active_global > 0:
            self._active_global -= 1
        if not any(item["lease_id"] == lease.lease_id and item["status"] in FINAL_STATUSES for item in ()):
            pass
        if lease not in self._history:
            self._history.append(lease)

    def _active_lease_counted(self, lease: Lease) -> bool:
        return lease.lease_id in self._leases and self._leases[lease.lease_id].status == ACTIVE

    async def _capacity_ready_locked(self, database_id: str) -> bool:
        database = self.spec.database(database_id)
        return (
            self._active_global < self.spec.limits.global_in_flight_leases
            and self._active_db[database_id] < database.lease_limit
            and self._inflight_db[database_id] < database.concurrency_quota
        )

    async def _acquire_slot(self, *, event_seq: int, database_id: str, batch_id: str) -> Lease:
        database = self.spec.database(database_id)
        if database.lease_limit == 0 or database.concurrency_quota == 0 or self.spec.limits.global_in_flight_leases == 0:
            raise RouteError(
                E_LEASE,
                f"databases[id={database_id}]",
                "configured lease/concurrency/global quota cannot admit a probe",
                {
                    "lease_limit": database.lease_limit,
                    "concurrency_quota": database.concurrency_quota,
                    "global_in_flight_leases": self.spec.limits.global_in_flight_leases,
                },
            )
        started_ns = time.monotonic_ns()
        wait_started_ms = started_ns // 1_000_000
        waited = False
        async with self._condition:
            while not await self._capacity_ready_locked(database_id):
                waited = True
                await self._condition.wait()
                if self._closed:
                    raise RouteError(E_LEASE, "scheduler", "scheduler is closed")
            wait_ended_ms = time.monotonic_ns() // 1_000_000
            self._active_global += 1
            self._active_db[database_id] += 1
            self._inflight_db[database_id] += 1
            self._peak_inflight_db[database_id] = max(
                self._peak_inflight_db[database_id], self._inflight_db[database_id]
            )
            lease_id = f"lease-{self.spec.digest[:16]}-{event_seq}-{database_id}"
            lease = Lease(
                lease_id=lease_id,
                database_id=database_id,
                event_seq=event_seq,
                route_version=self._route_version,
                batch_id=batch_id,
                created_ns=0,
                deadline_ns=0,
                lease_version=0,
                wait_started_ms=wait_started_ms,
                wait_ended_ms=wait_ended_ms,
                waited=waited,
            )
            self._leases[lease_id] = lease
            return lease

    async def _expire_due_locked(self, now_ns: int | None = None) -> None:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        changed = False
        for lease in self._leases.values():
            if lease.status == ACTIVE and lease.deadline_ns and now_ns >= lease.deadline_ns:
                lease.status = EXPIRED
                lease.decision_reason = "local_monotonic_deadline"
                lease.decision_event_seq = lease.event_seq
                lease.decision_route_version = lease.route_version
                self._release_active_locked(lease)
                changed = True
        if changed:
            self._condition.notify_all()

    async def _monitor_loop(self) -> None:
        interval = min(0.05, max(0.001, self.spec.limits.queue_poll_ms / 1000.0 / 2))
        while not self._closed:
            await asyncio.sleep(interval)
            async with self._condition:
                await self._expire_due_locked()

    async def _expire_due(self) -> None:
        async with self._condition:
            await self._expire_due_locked()

    async def invalidate_route_version(self, new_version: int, *, reason: str = "route_version_changed") -> list[dict[str, Any]]:
        async with self._condition:
            await self._expire_due_locked()
            invalidated: list[dict[str, Any]] = []
            for lease in self._leases.values():
                if lease.status == ACTIVE:
                    lease.status = VERSION_INVALIDATED
                    lease.decision_reason = reason
                    lease.decision_event_seq = lease.event_seq
                    lease.decision_route_version = new_version
                    self._release_active_locked(lease)
                    invalidated.append(lease.ownership())
            self._route_version = new_version
            self._condition.notify_all()
            return invalidated

    async def finalize_probe(self, lease: Lease, *, success: bool, reason: str | None = None) -> str:
        async with self._condition:
            now_ns = time.monotonic_ns()
            await self._expire_due_locked(now_ns)
            if lease.status == ACTIVE:
                if success and now_ns < lease.deadline_ns:
                    lease.status = COMPLETED
                    lease.decision_reason = "probe_completed_before_deadline"
                elif success:
                    lease.status = EXPIRED
                    lease.decision_reason = "local_monotonic_deadline"
                else:
                    lease.status = FAILED
                    lease.decision_reason = reason or "probe_failed"
                lease.decision_event_seq = lease.event_seq
                lease.decision_route_version = lease.route_version
                self._release_active_locked(lease)
            if not lease.inflight_released and self._inflight_db[lease.database_id] > 0:
                self._inflight_db[lease.database_id] -= 1
                lease.inflight_released = True
            self._condition.notify_all()
            return lease.status

    async def renew_lease(
        self,
        lease: Lease,
        *,
        ttl_ms: int,
        expected_lease_version: int | None = None,
    ) -> dict[str, Any]:
        async with self._condition:
            await self._expire_due_locked()
            if lease.status != ACTIVE:
                raise RouteError(
                    E_LEASE,
                    "lease.status",
                    "renewal rejected because the lease is no longer active",
                    {"lease_id": lease.lease_id, "status": lease.status},
                )
            expected = lease.lease_version if expected_lease_version is None else expected_lease_version
            if expected != lease.lease_version:
                raise RouteError(
                    E_LEASE,
                    "lease.lease_version",
                    "local optimistic lease version mismatch; deadline was not extended",
                    {"expected": lease.lease_version, "actual": expected, "lease_id": lease.lease_id},
                )
            database_id = lease.database_id
            prior_version = lease.lease_version
        worker = self.workers[database_id]
        response = await asyncio.to_thread(
            worker.renew_lease,
            lease_id=lease.lease_id,
            expected_lease_version=prior_version,
            route_version=lease.route_version,
            ttl_ms=ttl_ms,
        )
        async with self._condition:
            now_ns = time.monotonic_ns()
            await self._expire_due_locked(now_ns)
            if lease.status != ACTIVE or lease.lease_version != prior_version:
                raise RouteError(
                    E_LEASE,
                    "lease.lease_version",
                    "lease changed while renewing; deadline was not extended locally",
                    {"lease_id": lease.lease_id, "status": lease.status},
                )
            if now_ns >= lease.deadline_ns:
                lease.status = EXPIRED
                lease.decision_reason = "local_monotonic_deadline"
                lease.decision_event_seq = lease.event_seq
                lease.decision_route_version = lease.route_version
                self._release_active_locked(lease)
                self._condition.notify_all()
                raise RouteError(
                    E_LEASE,
                    "lease.deadline",
                    "lease expired while renewal acknowledgment was in flight",
                    {"lease_id": lease.lease_id},
                )
            lease.lease_version += 1
            lease.deadline_ns = now_ns + ttl_ms * 1_000_000
            return response

    async def _probe_job(self, job: dict[str, Any], batch_id: str) -> ProbeOutcome:
        event_seq = int(job["event_seq"])
        database_id = str(job["database_id"])
        key_record = {"seq": event_seq, "table": str(job["table"]), "key": int(job["key"])}
        lease = await self._acquire_slot(event_seq=event_seq, database_id=database_id, batch_id=batch_id)
        worker = self.workers[database_id]
        try:
            lease_ack = await asyncio.to_thread(
                worker.issue_lease,
                lease_id=lease.lease_id,
                event_seq=event_seq,
                route_version=lease.route_version,
                ttl_ms=self.spec.limits.lease_duration_ms,
            )
        except Exception:
            await self.finalize_probe(lease, success=False, reason="lease_issue_failed")
            raise
        async with self._condition:
            lease.lease_version = int(lease_ack["lease_version"])
            lease.created_ns = time.monotonic_ns()
            lease.deadline_ns = lease.created_ns + int(lease_ack["ttl_ms"]) * 1_000_000
        try:
            receipt = await asyncio.to_thread(
                worker.probe,
                batch_id=batch_id,
                keys=[key_record],
                route_version=lease.route_version,
                lease_id=lease.lease_id,
                lease_version=lease.lease_version,
            )
            expected = expected_probe_receipt(
                self.spec,
                batch_id=batch_id,
                database_id=database_id,
                keys=[key_record],
            )
            differences = first_field_difference(expected["results"], receipt.get("hits"))
            if differences is not None:
                raise RouteError(
                    E_REPLY,
                    f"reply.hits{differences}",
                    "worker route hits differ from deterministic parent expectation",
                    {"event_seq": event_seq},
                )
            if receipt.get("probe_digest") != expected["probe_digest"]:
                raise RouteError(
                    E_REPLY,
                    "reply.probe_digest",
                    "worker fixed-point receipt digest mismatch",
                    {"event_seq": event_seq},
                )
            if receipt.get("fixedpoint_checksum_u64") != expected["fixedpoint_checksum_u64"]:
                raise RouteError(
                    E_REPLY,
                    "reply.fixedpoint_checksum_u64",
                    "worker fixed-point integer checksum mismatch",
                    {"event_seq": event_seq},
                )
            status = await self.finalize_probe(lease, success=True)
            if status != COMPLETED:
                raise RouteError(
                    E_LEASE,
                    "lease.status",
                    "probe finished after lease expiry",
                    {"lease_id": lease.lease_id, "status": status, "event_seq": event_seq},
                )
        except Exception as exc:
            await self.finalize_probe(lease, success=False, reason="probe_failed")
            try:
                await asyncio.to_thread(
                    worker.release_lease,
                    lease_id=lease.lease_id,
                    lease_version=lease.lease_version,
                    outcome=FAILED,
                )
            except Exception:
                pass
            raise
        else:
            try:
                await asyncio.to_thread(
                    worker.release_lease,
                    lease_id=lease.lease_id,
                    lease_version=lease.lease_version,
                    outcome=COMPLETED,
                )
            except Exception:
                pass
            return ProbeOutcome(event_seq=event_seq, database_id=database_id, keys=[key_record], receipt=receipt, lease=lease)

    async def run_jobs(self, jobs: list[dict[str, Any]], *, batch_id: str) -> list[ProbeOutcome]:
        if not jobs:
            return []
        results = await asyncio.gather(*(self._probe_job(job, batch_id) for job in jobs), return_exceptions=True)
        outcomes: list[ProbeOutcome] = []
        error: BaseException | None = None
        for result in results:
            if isinstance(result, BaseException):
                if error is None:
                    error = result
            else:
                outcomes.append(result)
        if error is not None:
            raise error
        outcomes.sort(key=lambda outcome: outcome.event_seq)
        return outcomes

    def lease_history(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for lease in self._history:
            records.append(lease.to_dict())
        records.sort(key=lambda record: (record["event_seq"], record["database_id"]))
        return records

    def peak_in_flight(self) -> dict[str, int]:
        return dict(self._peak_inflight_db)

    def ownership_digest(self) -> str:
        return sha256_digest(self.lease_history_runtime_free())

    def lease_history_runtime_free(self) -> list[dict[str, Any]]:
        records = [lease.ownership() for lease in self._history]
        records.sort(key=lambda record: (record["event_seq"], record["database_id"]))
        return records


__all__ = [
    "ACTIVE",
    "COMPLETED",
    "EXPIRED",
    "FAILED",
    "FINAL_STATUSES",
    "Lease",
    "LeaseScheduler",
    "ProbeOutcome",
    "VERSION_INVALIDATED",
]

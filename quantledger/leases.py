"""Thread-safe lease state machine with monotonic generations and real reclamation."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import (
    E_BAD_LEASE_LIMIT,
    E_LEASE_CAPACITY,
    E_LEASE_EXPIRED,
    E_LEASE_NOT_FOUND,
    E_RECONCILE_PENDING,
    E_STALE_GENERATION,
    QLabError,
)


def default_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class Lease:
    lease_id: str
    generation: int
    batch_id: str
    revision: int
    worker_id: str
    issued_at_ms: int
    expires_at_ms: int
    status: str = "active"
    result: dict[str, Any] | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)

    def to_dict(self, *, now_ms: int | None = None) -> dict[str, Any]:
        payload = {
            "lease_id": self.lease_id,
            "generation": self.generation,
            "batch_id": self.batch_id,
            "revision": self.revision,
            "worker_id": self.worker_id,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "status": self.status,
            "has_result": self.result is not None,
        }
        if now_ms is not None:
            payload["now_ms"] = now_ms
        return payload


class LeaseBook:
    """Worker-local lease registry.

    Expiration is an explicit state transition: the lease leaves the active
    set, the lease-limit slot is reclaimed, a cancellation event is set, and an
    E_LEASE_EXPIRED event is queued. A compute thread cannot publish a result
    after that transition.
    """

    def __init__(
        self,
        *,
        worker_id: str,
        concurrency_quota: int,
        lease_limit: int,
        clock_ms: Callable[[], int] = default_clock_ms,
    ) -> None:
        if concurrency_quota < 1 or lease_limit < 1:
            raise QLabError(E_BAD_LEASE_LIMIT, "quota and lease limit must be positive")
        self.worker_id = worker_id
        self.concurrency_quota = int(concurrency_quota)
        self.lease_limit = int(lease_limit)
        self._clock_ms = clock_ms
        self._lock = threading.RLock()
        self._leases: dict[str, Lease] = {}
        self._active: set[str] = set()
        self._running: set[str] = set()
        self._next_generation = 1
        self._events: list[dict[str, Any]] = []
        self._expired_tombstones: set[str] = set()
        self._closed = False

    def _now(self) -> int:
        return int(self._clock_ms())

    def _new_event(self, code: str, **payload: Any) -> None:
        event = {
            "code": code,
            "worker_id": self.worker_id,
            "event_seq": len(self._events) + 1,
            **payload,
        }
        self._events.append(event)

    def grant(self, *, batch_id: str, revision: int, ttl_ms: int) -> Lease:
        if ttl_ms <= 0:
            raise QLabError(E_LEASE_CAPACITY, "ttl_ms must be positive", code_detail=E_LEASE_CAPACITY)
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            if len(self._active) >= self.lease_limit:
                raise QLabError(
                    E_LEASE_CAPACITY,
                    "lease limit reached",
                    active_leases=len(self._active),
                    lease_limit=self.lease_limit,
                )
            generation = self._next_generation
            self._next_generation += 1
            lease_id = f"{self.worker_id}:{generation}"
            lease = Lease(
                lease_id=lease_id,
                generation=generation,
                batch_id=batch_id,
                revision=int(revision),
                worker_id=self.worker_id,
                issued_at_ms=now,
                expires_at_ms=now + int(ttl_ms),
            )
            self._leases[lease_id] = lease
            self._active.add(lease_id)
            return lease

    def _require(self, lease_id: str) -> Lease:
        lease = self._leases.get(lease_id)
        if lease is None:
            raise QLabError(E_LEASE_NOT_FOUND, "lease not found", lease_id=lease_id)
        if lease.status in {"expired", "released"}:
            code = E_LEASE_EXPIRED if lease.status == "expired" else E_LEASE_EXPIRED
            raise QLabError(
                code,
                "lease is no longer active",
                lease_id=lease_id,
                generation=lease.generation,
                status=lease.status,
                generation_unchanged=True,
            )
        return lease

    def _check_generation(self, lease: Lease, generation: int, revision: int | None = None) -> None:
        if lease.generation != int(generation):
            raise QLabError(
                E_STALE_GENERATION,
                "stale lease generation",
                lease_id=lease.lease_id,
                expected_generation=lease.generation,
                received_generation=int(generation),
                generation_unchanged=True,
            )
        if revision is not None and lease.revision != int(revision):
            raise QLabError(
                E_STALE_GENERATION,
                "lease revision no longer matches",
                lease_id=lease.lease_id,
                expected_revision=lease.revision,
                received_revision=int(revision),
                generation_unchanged=True,
            )

    def renew(self, *, lease_id: str, generation: int, ttl_ms: int, revision: int | None = None) -> Lease:
        if ttl_ms <= 0:
            raise QLabError(E_LEASE_CAPACITY, "ttl_ms must be positive")
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            lease = self._require(lease_id)
            self._check_generation(lease, generation, revision)
            lease.expires_at_ms = now + int(ttl_ms)
            lease.status = "renewed" if lease.status == "active" else lease.status
            return lease

    def release(self, *, lease_id: str, generation: int, revision: int | None = None) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            lease = self._require(lease_id)
            self._check_generation(lease, generation, revision)
            lease.status = "released"
            lease.cancelled.set()
            self._active.discard(lease_id)
            self._running.discard(lease_id)
            return {
                "lease_id": lease_id,
                "generation": lease.generation,
                "status": "released",
                "active_leases": len(self._active),
                "running": len(self._running),
                "released_at_ms": now,
            }

    def begin_run(self, lease_id: str, generation: int) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.status in {"expired", "released"} or lease.generation != generation:
                return False
            self._running.add(lease_id)
            lease.status = "running"
            return True

    def finish_run(self, lease_id: str, generation: int, result: dict[str, Any]) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            self._running.discard(lease_id)
            if lease is None or lease.status in {"expired", "released"} or lease.generation != generation:
                if lease is not None:
                    self._new_event(
                        E_RECONCILE_PENDING,
                        lease_id=lease_id,
                        generation=generation,
                        batch_id=lease.batch_id,
                        reason="result_arrived_after_lease_loss",
                    )
                return False
            lease.result = copy.deepcopy(result)
            lease.status = "completed"
            return True

    def fetch(self, lease_id: str) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            lease = self._leases.get(lease_id)
            if lease is None:
                raise QLabError(E_LEASE_NOT_FOUND, "lease not found", lease_id=lease_id)
            if lease.status == "expired":
                raise QLabError(
                    E_RECONCILE_PENDING,
                    "lease expired before result was collected",
                    lease_id=lease_id,
                    generation=lease.generation,
                    batch_id=lease.batch_id,
                )
            if lease.status == "released":
                raise QLabError(E_LEASE_EXPIRED, "lease was released", lease_id=lease_id)
            if lease.status in {"active", "renewed"}:
                return {"status": "running", "lease": lease.to_dict(now_ms=now)}
            if lease.status == "running":
                return {"status": "running", "lease": lease.to_dict(now_ms=now)}
            return {"status": "completed", "lease": lease.to_dict(now_ms=now), "result": copy.deepcopy(lease.result)}

    def expire(self) -> list[dict[str, Any]]:
        with self._lock:
            before = len(self._events)
            self._expire_locked(self._now())
            return [copy.deepcopy(item) for item in self._events[before:]]

    def _expire_locked(self, now: int) -> None:
        expired = [lease_id for lease_id in self._active if self._leases[lease_id].expires_at_ms <= now]
        for lease_id in sorted(expired):
            lease = self._leases[lease_id]
            lease.status = "expired"
            lease.cancelled.set()
            self._active.discard(lease_id)
            self._expired_tombstones.add(lease_id)
            self._new_event(
                E_LEASE_EXPIRED,
                lease_id=lease_id,
                generation=lease.generation,
                batch_id=lease.batch_id,
                revision=lease.revision,
                expires_at_ms=lease.expires_at_ms,
                reclaimed_at_ms=now,
                reclaimed_units=1,
                active_after=len(self._active),
                running_after=len(self._running),
            )

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = copy.deepcopy(self._events)
            self._events.clear()
            return events

    def set_lease_limit(self, value: int) -> dict[str, Any]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise QLabError(E_BAD_LEASE_LIMIT, "lease_limit must be a positive integer")
        with self._lock:
            self.lease_limit = int(value)
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._now()
            self._expire_locked(now)
            return {
                "worker_id": self.worker_id,
                "concurrency_quota": self.concurrency_quota,
                "lease_limit": self.lease_limit,
                "active_leases": len(self._active),
                "running": len(self._running),
                "next_generation": self._next_generation,
                "now_ms": now,
                "active": [self._leases[lease_id].to_dict(now_ms=now) for lease_id in sorted(self._active)],
            }

    def get(self, lease_id: str) -> Lease | None:
        with self._lock:
            return self._leases.get(lease_id)

    def stop(self) -> None:
        with self._lock:
            now = self._now()
            for lease_id in list(self._active):
                lease = self._leases[lease_id]
                lease.status = "released"
                lease.cancelled.set()
                lease.expires_at_ms = now
            self._active.clear()
            self._running.clear()
            self._closed = True


class LeaseReaper(threading.Thread):
    """Small explicit sweeper; all resource counters are changed under a lock."""

    def __init__(self, book: LeaseBook, interval_ms: int = 5) -> None:
        super().__init__(name=f"lease-reaper-{book.worker_id}", daemon=True)
        self.book = book
        self.interval_ms = max(1, int(interval_ms))
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_ms / 1000.0):
            self.book.expire()

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=1.0)


"""Local executor leases with expiry, generations and real unit reclamation."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Any

from .errors import BusinessError, E_LEASE_EXPIRED, E_LEASE_GENERATION


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class ManualClock:
    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def __call__(self) -> int:
        return self._now

    def advance(self, milliseconds: int) -> int:
        if milliseconds < 0:
            raise ValueError("cannot move time backwards")
        self._now += int(milliseconds)
        return self._now


@dataclass
class Lease:
    lease_id: str
    generation: int
    steps: tuple[int, ...]
    issued_ms: int
    expires_ms: int
    released: bool = False
    release_reason: str | None = None
    renewed_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "steps": list(self.steps),
        }


@dataclass
class LeaseEvent:
    code: str
    generation: int
    lease_id: str | None
    steps: tuple[int, ...]
    at_ms: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "generation": self.generation,
            "lease_id": self.lease_id,
            "steps": list(self.steps),
            "at_ms": self.at_ms,
            **self.detail,
        }


class LocalExecutor:
    """Owns leased work units and reclaims them eagerly on expiry checks."""

    def __init__(self, clock: Callable[[], int] | None = None) -> None:
        self._clock = clock or monotonic_ms
        self._lock = threading.RLock()
        self._generation = 0
        self._next_lease_number = 1
        self._leases: dict[str, Lease] = {}
        self._occupied: set[tuple[str, int]] = set()
        self._events: list[LeaseEvent] = []

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def now_ms(self) -> int:
        return int(self._clock())

    def acquire(self, steps: Iterable[int], ttl_ms: int) -> Lease:
        if ttl_ms <= 0:
            raise BusinessError("E_LEASE_CONFIG", "lease ttl must be positive", {"ttl_ms": ttl_ms})
        step_tuple = tuple(sorted({int(step) for step in steps}))
        if not step_tuple or step_tuple[0] < 0:
            raise BusinessError("E_LEASE_TARGET", "lease must cover at least one non-negative work step")
        with self._lock:
            now = self.now_ms()
            self._reclaim_expired_locked(now)
            self._generation += 1
            generation = self._generation
            lease_id = f"lease-{generation:06d}-{self._next_lease_number:04d}"
            self._next_lease_number += 1
            lease = Lease(
                lease_id=lease_id,
                generation=generation,
                steps=step_tuple,
                issued_ms=now,
                expires_ms=now + ttl_ms,
            )
            self._leases[lease_id] = lease
            self._occupied.update((lease_id, step) for step in step_tuple)
            return Lease(**lease.as_dict() | {"steps": lease.steps})

    def _reclaim_expired_locked(self, now_ms: int) -> list[LeaseEvent]:
        expired = [lease for lease in self._leases.values()
                   if not lease.released and now_ms >= lease.expires_ms]
        events: list[LeaseEvent] = []
        for lease in expired:
            self._release_locked(lease, now_ms, "expired")
            event = LeaseEvent(
                code=E_LEASE_EXPIRED,
                generation=lease.generation,
                lease_id=lease.lease_id,
                steps=lease.steps,
                at_ms=now_ms,
                detail={"expired_ms": lease.expires_ms,
                        "reclaimed_units": len(lease.steps)},
            )
            events.append(event)
            self._events.append(event)
        return events

    def reclaim_expired(self, now_ms: int | None = None) -> list[LeaseEvent]:
        with self._lock:
            return self._reclaim_expired_locked(self.now_ms() if now_ms is None else int(now_ms))

    def _find_current(self, lease_id: str, generation: int, now_ms: int,
                      *, reclaim_if_expired: bool) -> Lease:
        lease = self._leases.get(lease_id)
        if lease is None or lease.generation != generation:
            raise BusinessError(
                E_LEASE_GENERATION,
                "lease generation does not match the acknowledged generation",
                {"lease_id": lease_id, "expected_generation": None if lease is None else lease.generation,
                 "provided_generation": generation, "current_executor_generation": self._generation},
            )
        if lease.released:
            raise BusinessError(
                E_LEASE_EXPIRED,
                "lease has already been released or reclaimed",
                {"lease_id": lease_id, "generation": generation, "reason": lease.release_reason},
            )
        if now_ms >= lease.expires_ms:
            if reclaim_if_expired:
                self._reclaim_expired_locked(now_ms)
            raise BusinessError(
                E_LEASE_EXPIRED,
                "lease expired before the operation; occupied units were reclaimed",
                {"lease_id": lease_id, "generation": generation, "expires_ms": lease.expires_ms,
                 "now_ms": now_ms, "reclaimed_units": len(lease.steps)},
            )
        return lease

    def renew(self, lease_id: str, generation: int, ttl_ms: int | None = None) -> Lease:
        if ttl_ms is None:
            raise BusinessError("E_LEASE_CONFIG", "renew requires an explicit ttl_ms")
        if ttl_ms <= 0:
            raise BusinessError("E_LEASE_CONFIG", "lease ttl must be positive", {"ttl_ms": ttl_ms})
        with self._lock:
            now = self.now_ms()
            self._reclaim_expired_locked(now)
            lease = self._find_current(lease_id, generation, now, reclaim_if_expired=False)
            lease.expires_ms = now + ttl_ms
            lease.renewed_count += 1
            return Lease(**lease.as_dict() | {"steps": lease.steps})

    def release(self, lease_id: str, generation: int) -> dict[str, Any]:
        with self._lock:
            now = self.now_ms()
            self._reclaim_expired_locked(now)
            lease = self._find_current(lease_id, generation, now, reclaim_if_expired=False)
            self._release_locked(lease, now, "released")
            event = LeaseEvent(
                code="LEASE_RELEASED",
                generation=lease.generation,
                lease_id=lease.lease_id,
                steps=lease.steps,
                at_ms=now,
                detail={"reclaimed_units": len(lease.steps)},
            )
            self._events.append(event)
            return event.as_dict()

    def assert_active(self, lease_id: str, generation: int, step: int) -> None:
        with self._lock:
            now = self.now_ms()
            self._reclaim_expired_locked(now)
            lease = self._find_current(lease_id, generation, now, reclaim_if_expired=False)
            if step not in lease.steps:
                raise BusinessError("E_LEASE_TARGET", "step is not covered by the lease",
                                    {"lease_id": lease_id, "step": step, "steps": list(lease.steps)})
            if (lease_id, step) not in self._occupied:
                raise BusinessError("E_LEASE_TARGET", "leased work unit is no longer occupied",
                                    {"lease_id": lease_id, "step": step})

    def _release_locked(self, lease: Lease, at_ms: int, reason: str) -> None:
        lease.released = True
        lease.release_reason = reason
        for step in lease.steps:
            self._occupied.discard((lease.lease_id, step))
        # Keep the lease tombstone for generation diagnostics, but never expose
        # its work units as occupied.

    def occupied_units(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{"lease_id": lease_id, "step": step}
                    for lease_id, step in sorted(self._occupied)]

    def leases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [lease.as_dict() for lease in sorted(self._leases.values(),
                                                        key=lambda item: item.generation)]

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.as_dict() for event in self._events]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "occupied_units": self.occupied_units(),
                "leases": self.leases(),
                "events": self.events(),
            }

    def close(self) -> list[dict[str, Any]]:
        """Release every occupied unit without delaying; useful for test teardown."""
        with self._lock:
            now = self.now_ms()
            released = []
            for lease in self._leases.values():
                if not lease.released:
                    self._release_locked(lease, now, "closed")
                    released.append(lease.as_dict())
            return released

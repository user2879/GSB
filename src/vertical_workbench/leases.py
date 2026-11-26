from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .clock import Clock, MonotonicClock
from .codes import OK, E_LEASE_EXPIRED, E_LEASE_GENERATION, E_LEASE_LIMIT


@dataclass
class LeaseRecord:
    lease_id: str
    target: str
    x_min: int
    x_max: int
    generation: int
    issued_ms: int
    expires_ms: int
    reserved_points: int
    status: str = "ACTIVE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseExecutor:
    """In-process local executor.

    All mutations are locked.  A wrong generation is a compare-and-swap failure
    and deliberately leaves the accepted generation untouched.
    """

    def __init__(self, clock: Clock | None = None, max_active_leases: int = 12):
        self.clock = clock or MonotonicClock()
        self.max_active_leases = max_active_leases
        self._lock = threading.RLock()
        self._leases: dict[str, LeaseRecord] = {}
        self._active_by_target: dict[str, str] = {}
        self._generation_by_target: dict[str, int] = {}
        self._generation = 0
        self.reserved_points = 0
        self.events: list[dict[str, Any]] = []

    def _event_locked(self, code: str, lease: LeaseRecord | None, **extra: Any) -> dict[str, Any]:
        event = {
            "code": code,
            "at_ms": self.clock.now_ms(),
            "generation": lease.generation if lease else self._generation,
            "lease_id": lease.lease_id if lease else None,
            "target": lease.target if lease else None,
        }
        event.update(extra)
        self.events.append(event)
        return event

    def reap_expired(self) -> list[dict[str, Any]]:
        now = self.clock.now_ms()
        expired_events: list[dict[str, Any]] = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.status == "ACTIVE" and lease.expires_ms <= now:
                    lease.status = "EXPIRED"
                    self._active_by_target.pop(lease.target, None)
                    self.reserved_points = max(0, self.reserved_points - lease.reserved_points)
                    event = self._event_locked(
                        E_LEASE_EXPIRED,
                        lease,
                        expired_at_ms=lease.expires_ms,
                        reaped_at_ms=now,
                        released_points=lease.reserved_points,
                    )
                    expired_events.append(event)
        return expired_events

    def acquire(
        self,
        target: str,
        x_min: int,
        x_max: int,
        reserved_points: int = 0,
        ttl_ms: int = 60_000,
    ) -> dict[str, Any]:
        with self._lock:
            self.reap_expired()
            if target in self._active_by_target:
                existing = self._leases[self._active_by_target[target]]
                return {"ok": False, "code": "E_LEASE_CONFLICT", "lease": existing.to_dict()}
            active = [lease for lease in self._leases.values() if lease.status == "ACTIVE"]
            if len(active) >= self.max_active_leases:
                event = self._event_locked(E_LEASE_LIMIT, None, target=target)
                return {"ok": False, "code": E_LEASE_LIMIT, "event": event}
            for lease in active:
                if lease.target.split(":c", 1)[0] != target.split(":c", 1)[0]:
                    continue
                if x_min < lease.x_max and lease.x_min < x_max:
                    return {"ok": False, "code": "E_LEASE_OVERLAP", "lease": lease.to_dict()}

            now = self.clock.now_ms()
            target_generation = self._generation_by_target.get(target, 0) + 1
            self._generation_by_target[target] = target_generation
            self._generation += 1
            lease = LeaseRecord(
                lease_id=f"lease-{uuid.uuid5(uuid.NAMESPACE_URL, target + ':' + str(target_generation)).hex[:16]}",
                target=target,
                x_min=int(x_min),
                x_max=int(x_max),
                generation=target_generation,
                issued_ms=now,
                expires_ms=now + int(ttl_ms),
                reserved_points=int(reserved_points),
            )
            self._leases[lease.lease_id] = lease
            self._active_by_target[target] = lease.lease_id
            self.reserved_points += lease.reserved_points
            return {"ok": True, "code": OK, "lease": lease.to_dict()}

    def renew(self, lease_id: str, generation: int, ttl_ms: int | None = None) -> dict[str, Any]:
        with self._lock:
            self.reap_expired()
            lease = self._leases.get(lease_id)
            if lease is None:
                return {"ok": False, "code": E_LEASE_EXPIRED}
            if lease.status != "ACTIVE":
                return {"ok": False, "code": E_LEASE_EXPIRED, "lease": lease.to_dict()}
            if generation != lease.generation:
                before = self._generation
                event = self._event_locked(
                    E_LEASE_GENERATION,
                    lease,
                    presented_generation=generation,
                    accepted_generation=lease.generation,
                )
                assert self._generation == before
                return {"ok": False, "code": E_LEASE_GENERATION, "event": event}

            now = self.clock.now_ms()
            lease.expires_ms = now + int(ttl_ms or max(1, lease.expires_ms - lease.issued_ms))
            self._event_locked("LEASE_RENEWED", lease, expires_ms=lease.expires_ms)
            return {"ok": True, "code": OK, "lease": lease.to_dict()}

    def release(self, lease_id: str, generation: int) -> dict[str, Any]:
        with self._lock:
            self.reap_expired()
            lease = self._leases.get(lease_id)
            if lease is None or lease.status != "ACTIVE":
                return {"ok": False, "code": E_LEASE_EXPIRED}
            if generation != lease.generation:
                before = self._generation
                event = self._event_locked(
                    E_LEASE_GENERATION,
                    lease,
                    presented_generation=generation,
                    accepted_generation=lease.generation,
                )
                assert self._generation == before
                return {"ok": False, "code": E_LEASE_GENERATION, "event": event}
            lease.status = "RELEASED"
            self._active_by_target.pop(lease.target, None)
            self.reserved_points = max(0, self.reserved_points - lease.reserved_points)
            self._event_locked("LEASE_RELEASED", lease)
            return {"ok": True, "code": OK, "lease": lease.to_dict()}

    def active_leases(self) -> list[dict[str, Any]]:
        with self._lock:
            self.reap_expired()
            return [
                lease.to_dict()
                for lease in sorted(self._leases.values(), key=lambda item: item.target)
                if lease.status == "ACTIVE"
            ]

    def active_lease_for_target(self, target: str) -> dict[str, Any] | None:
        with self._lock:
            self.reap_expired()
            lease_id = self._active_by_target.get(target)
            if not lease_id:
                return None
            lease = self._leases.get(lease_id)
            return lease.to_dict() if lease and lease.status == "ACTIVE" else None

    def lease_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [lease.to_dict() for lease in sorted(self._leases.values(), key=lambda item: item.target)]

    @property
    def current_generation(self) -> int:
        with self._lock:
            return self._generation

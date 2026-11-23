"""Correlation leases with optimistic renewal and real expiry reaping."""

import threading
from dataclasses import dataclass, field

from .clock import MonotonicClock
from .errors import LeaseError


@dataclass(frozen=True)
class BatchRange:
    service_id: str
    index_lo: int
    index_hi: int

    def to_dict(self) -> dict:
        return {
            "service_id": self.service_id,
            "index_lo": self.index_lo,
            "index_hi": self.index_hi,
        }


@dataclass
class Lease:
    lease_id: str
    service_id: str
    batch_range: BatchRange
    holder: str
    version: int
    granted_ms: int
    expires_ms: int
    reaped_ms: int | None = None

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_ms

    def to_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "service_id": self.service_id,
            "batch_range": self.batch_range.to_dict(),
            "holder": self.holder,
            "version": self.version,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "reaped_ms": self.reaped_ms,
        }


@dataclass
class LeaseEvent:
    lease_id: str
    service_id: str
    kind: str  # GRANT | RENEW | REAP | RELEASE
    at_ms: int
    version: int

    def to_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "service_id": self.service_id,
            "kind": self.kind,
            "at_ms": self.at_ms,
            "version": self.version,
        }


@dataclass
class LeaseOccupancy:
    """Deterministic occupancy snapshot for ledger persistence."""

    peak_global_active: int
    peak_per_service: dict[str, int]
    granted: int
    renewed: int
    reaped: int
    rejected: int
    events: list[LeaseEvent] = field(default_factory=list)

    def deterministic_dict(self) -> dict:
        return {
            "peak_global_active": self.peak_global_active,
            "peak_per_service": dict(sorted(self.peak_per_service.items())),
            "granted": self.granted,
            "renewed": self.renewed,
            "reaped": self.reaped,
            "rejected": self.rejected,
            "events": [event.to_dict() for event in self.events],
        }


class LeaseManager:
    def __init__(self, global_capacity: int, service_capacity: int,
                 ttl_ms: int, clock=None):
        if global_capacity < 1 or service_capacity < 1 or ttl_ms < 1:
            raise LeaseError("lease capacities and ttl must be positive",
                             field="lease_config")
        self.global_capacity = global_capacity
        self.service_capacity = service_capacity
        self.ttl_ms = ttl_ms
        self.clock = clock or MonotonicClock()
        self._lock = threading.RLock()
        self._active: dict[str, Lease] = {}
        self._per_service: dict[str, set[str]] = {}
        self._seq = 0
        self.occupancy = LeaseOccupancy(
            peak_global_active=0, peak_per_service={},
            granted=0, renewed=0, reaped=0, rejected=0, events=[])

    def _record(self, lease: Lease, kind: str) -> None:
        self.occupancy.events.append(LeaseEvent(
            lease_id=lease.lease_id, service_id=lease.service_id,
            kind=kind, at_ms=self.clock.now_ms(), version=lease.version))

    def active_count(self) -> int:
        with self._lock:
            self._reap_expired_locked()
            return len(self._active)

    def service_active_count(self, service_id: str) -> int:
        with self._lock:
            self._reap_expired_locked()
            return len(self._per_service.get(service_id, ()))

    def can_grant(self, service_id: str) -> bool:
        with self._lock:
            self._reap_expired_locked()
            if len(self._active) >= self.global_capacity:
                return False
            return len(self._per_service.get(service_id, ())) < self.service_capacity

    def grant(self, batch_range: BatchRange, holder: str) -> Lease:
        with self._lock:
            self._reap_expired_locked()
            now_ms = self.clock.now_ms()
            if len(self._active) >= self.global_capacity:
                self.occupancy.rejected += 1
                raise LeaseError(
                    f"global concurrency cap {self.global_capacity} reached",
                    field="global_concurrency")
            per_service = self._per_service.setdefault(batch_range.service_id, set())
            if len(per_service) >= self.service_capacity:
                self.occupancy.rejected += 1
                raise LeaseError(
                    f"service concurrency cap {self.service_capacity} reached "
                    f"for {batch_range.service_id}",
                    field="service_concurrency")
            self._seq += 1
            lease = Lease(
                lease_id=f"lease-{self._seq:08d}",
                service_id=batch_range.service_id,
                batch_range=batch_range,
                holder=holder,
                version=1,
                granted_ms=now_ms,
                expires_ms=now_ms + self.ttl_ms,
            )
            self._active[lease.lease_id] = lease
            per_service.add(lease.lease_id)
            self.occupancy.granted += 1
            self.occupancy.peak_global_active = max(
                self.occupancy.peak_global_active, len(self._active))
            self.occupancy.peak_per_service[lease.service_id] = max(
                self.occupancy.peak_per_service.get(lease.service_id, 0),
                len(per_service))
            self._record(lease, "GRANT")
            return lease

    def renew(self, lease_id: str, holder: str, expected_version: int,
              extension_ms: int | None = None) -> Lease:
        with self._lock:
            self._reap_expired_locked()
            lease = self._active.get(lease_id)
            if lease is None:
                self.occupancy.rejected += 1
                raise LeaseError(
                    f"lease {lease_id} is not active (expired or unknown)",
                    field="lease_id")
            if lease.holder != holder:
                self.occupancy.rejected += 1
                raise LeaseError(
                    f"lease {lease_id} held by {lease.holder!r}, "
                    f"renewal by {holder!r} rejected",
                    field="holder")
            if lease.version != expected_version:
                self.occupancy.rejected += 1
                raise LeaseError(
                    f"optimistic version conflict: expected {lease.version}, "
                    f"got {expected_version}",
                    field="version")
            now_ms = self.clock.now_ms()
            if now_ms >= lease.expires_ms:
                self.occupancy.rejected += 1
                raise LeaseError(
                    f"lease {lease_id} already expired at {lease.expires_ms}",
                    field="expires_ms")
            extension = self.ttl_ms if extension_ms is None else extension_ms
            lease.version += 1
            lease.expires_ms = now_ms + extension
            self.occupancy.renewed += 1
            self._record(lease, "RENEW")
            return lease

    def release(self, lease_id: str) -> None:
        with self._lock:
            lease = self._active.pop(lease_id, None)
            if lease is None:
                return
            self._per_service.get(lease.service_id, set()).discard(lease_id)
            self._record(lease, "RELEASE")

    def reap_expired(self) -> list[Lease]:
        with self._lock:
            return self._reap_expired_locked()

    def _reap_expired_locked(self) -> list[Lease]:
        now_ms = self.clock.now_ms()
        expired = [lease for lease in self._active.values()
                   if lease.is_expired(now_ms)]
        for lease in expired:
            lease.reaped_ms = now_ms
            self._per_service.get(lease.service_id, set()).discard(lease.lease_id)
            del self._active[lease.lease_id]
            self.occupancy.reaped += 1
            self._record(lease, "REAP")
        return expired

    def snapshot_deterministic(self) -> dict:
        with self._lock:
            return self.occupancy.deterministic_dict()

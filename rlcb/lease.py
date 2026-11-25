"""Concurrency leases and unique-ownership arbitration.

A lease grants one concurrency slot of one service to one request batch.
Leases carry an explicit version used for optimistic renewal: renewing with a
stale version fails with ``E_LEASE`` and never extends the deadline.

Expiry is decided against a real monotonic clock (no artificial sleeps). The
arbitration logic is also exposed as a pure, event-sorted function so that any
reordering of the recorded events reproduces the identical ownership
conclusion (the ledger stores events in canonical event-sequence order).

Two reclamation reasons must never double-count a slot at the same instant:

* ``expired``   - the lease deadline passed on the monotonic clock;
* ``superseded``- a quota-version generation change invalidated the lease.

Each slot therefore has a single winning event per change, chosen by a total
order on (event_seq, reason, lease_id) and recorded with its basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any

from .clock import MonotonicClock
from .codes import LeaseError

REASON_GRANT = "grant"
REASON_RENEW = "renew"
REASON_EXPIRED = "expired"
REASON_SUPERSEDED = "superseded"
REASON_RELEASED = "released"


@dataclass(frozen=True)
class Lease:
    lease_id: str
    service_id: str
    batch_id: str
    granted_ms: int
    expires_ms: int
    version: int
    quota_version: int
    slot_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "service_id": self.service_id,
            "batch_id": self.batch_id,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "version": self.version,
            "quota_version": self.quota_version,
            "slot_index": self.slot_index,
        }


@dataclass(frozen=True)
class LeaseEvent:
    event_seq: int
    event_type: str
    at_ms: int
    lease_id: str
    service_id: str
    batch_id: str
    slot_index: int
    version: int
    quota_version: int
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "event_type": self.event_type,
            "at_ms": self.at_ms,
            "lease_id": self.lease_id,
            "service_id": self.service_id,
            "batch_id": self.batch_id,
            "slot_index": self.slot_index,
            "version": self.version,
            "quota_version": self.quota_version,
            "basis": self.basis,
        }


class LeaseRegistry:
    """Authoritative per-service lease store used by the service subprocess."""

    def __init__(self, service_id: str, ttl_ms: int):
        self.service_id = service_id
        self.ttl_ms = ttl_ms
        self._versions: dict[str, int] = {}

    def check_version(self, lease_id: str, expected_version: int) -> bool:
        return self._versions.get(lease_id) == expected_version

    def record_grant(self, lease_id: str, version: int) -> None:
        self._versions[lease_id] = version

    def bump(self, lease_id: str) -> int:
        new_version = self._versions.get(lease_id, 0) + 1
        self._versions[lease_id] = new_version
        return new_version


class LeaseManager:
    """Main-side manager: real-clock grants, renewals and arbitration."""

    def __init__(self, clock: MonotonicClock, ttl_ms: int):
        self.clock = clock
        self.ttl_ms = int(ttl_ms)
        self._seq = count()
        self.events: list[LeaseEvent] = []
        # (service_id, slot_index) -> active lease
        self.active: dict[tuple[str, int], Lease] = {}
        # service_id -> set of occupied slot indexes
        self._occupied: dict[str, set[int]] = {}
        self._leases: dict[str, Lease] = {}

    def _next_event(self, event_type: str, at_ms: int, lease: Lease,
                    basis: str) -> LeaseEvent:
        event = LeaseEvent(
            event_seq=next(self._seq),
            event_type=event_type,
            at_ms=at_ms,
            lease_id=lease.lease_id,
            service_id=lease.service_id,
            batch_id=lease.batch_id,
            slot_index=lease.slot_index,
            version=lease.version,
            quota_version=lease.quota_version,
            basis=basis,
        )
        self.events.append(event)
        return event

    def grant(self, *, service_id: str, batch_id: str, slot_index: int,
              quota_version: int, lease_seq: int) -> Lease:
        key = (service_id, slot_index)
        if key in self.active:
            raise LeaseError(
                f"slot {slot_index} of {service_id} already held",
                service_id=service_id, slot_index=slot_index,
                holder=self.active[key].lease_id,
            )
        now = self.clock.now_ms()
        lease = Lease(
            lease_id=f"LS-{service_id}-{lease_seq:05d}",
            service_id=service_id,
            batch_id=batch_id,
            granted_ms=now,
            expires_ms=now + self.ttl_ms,
            version=1,
            quota_version=quota_version,
            slot_index=slot_index,
        )
        self.active[key] = lease
        self._occupied.setdefault(service_id, set()).add(slot_index)
        self._leases[lease.lease_id] = lease
        self._next_event(REASON_GRANT, now, lease,
                         f"grant_slot_{slot_index}_qv{quota_version}")
        return lease

    def is_expired(self, lease: Lease) -> bool:
        return self.clock.now_ms() >= lease.expires_ms

    def renew(self, lease_id: str, expected_version: int,
              registry: LeaseRegistry) -> Lease:
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError(f"unknown lease {lease_id}", lease_id=lease_id)
        # Optimistic check against the subprocess authoritative versions.
        if not registry.check_version(lease_id, expected_version):
            raise LeaseError(
                f"stale lease version for {lease_id}: expected server version,"
                f" got {expected_version}",
                lease_id=lease_id, expected_version=expected_version,
                server_version=registry._versions.get(lease_id),
            )
        if expected_version != lease.version:
            raise LeaseError(
                f"stale lease version for {lease_id}",
                lease_id=lease_id, expected_version=expected_version,
                current_version=lease.version,
            )
        if self.is_expired(lease):
            raise LeaseError(
                f"lease {lease_id} already expired; renewal refused",
                lease_id=lease_id, expires_ms=lease.expires_ms,
                now_ms=self.clock.now_ms(),
            )
        new_version = registry.bump(lease_id)
        now = self.clock.now_ms()
        renewed = Lease(
            lease_id=lease.lease_id,
            service_id=lease.service_id,
            batch_id=lease.batch_id,
            granted_ms=lease.granted_ms,
            expires_ms=now + self.ttl_ms,
            version=new_version,
            quota_version=lease.quota_version,
            slot_index=lease.slot_index,
        )
        self._leases[lease_id] = renewed
        self.active[(lease.service_id, lease.slot_index)] = renewed
        self._next_event(REASON_RENEW, now, renewed,
                         f"renew_optimistic_v{expected_version}_to_v{new_version}")
        return renewed

    def release(self, lease_id: str) -> LeaseEvent | None:
        """Normal completion release; no-op if already reclaimed."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return None
        key = (lease.service_id, lease.slot_index)
        current = self.active.get(key)
        if current is None or current.lease_id != lease_id:
            return None
        return self._reclaim(lease, REASON_RELEASED, self.clock.now_ms(),
                             "batch_completed_normal_release")

    def reclaim_expired(self) -> list[LeaseEvent]:
        now = self.clock.now_ms()
        reclaimed: list[LeaseEvent] = []
        for key in sorted(self.active):
            lease = self.active[key]
            if now >= lease.expires_ms:
                event = self._reclaim(lease, REASON_EXPIRED, now,
                                      f"clock_{now}>=expiry_{lease.expires_ms}")
                reclaimed.append(event)
        return reclaimed

    def reclaim_superseded(self, new_quota_version: int) -> list[LeaseEvent]:
        now = self.clock.now_ms()
        reclaimed: list[LeaseEvent] = []
        for key in sorted(self.active):
            lease = self.active[key]
            if lease.quota_version < new_quota_version:
                reason = (REASON_EXPIRED if now >= lease.expires_ms
                          else REASON_SUPERSEDED)
                basis = (f"clock_{now}>=expiry_{lease.expires_ms}"
                         if reason == REASON_EXPIRED
                         else f"qv_{lease.quota_version}<{new_quota_version}")
                event = self._reclaim(lease, reason, now, basis)
                reclaimed.append(event)
        return reclaimed

    def _reclaim(self, lease: Lease, reason: str, at_ms: int,
                 basis: str) -> LeaseEvent:
        self.active.pop((lease.service_id, lease.slot_index), None)
        self._occupied.get(lease.service_id, set()).discard(lease.slot_index)
        tombstone = Lease(
            lease_id=lease.lease_id,
            service_id=lease.service_id,
            batch_id=lease.batch_id,
            granted_ms=lease.granted_ms,
            expires_ms=lease.expires_ms,
            version=lease.version,
            quota_version=lease.quota_version,
            slot_index=lease.slot_index,
        )
        return self._next_event(reason, at_ms, tombstone, basis)

    def active_count(self, service_id: str | None = None) -> int:
        if service_id is None:
            return len(self.active)
        return sum(1 for (sid, _) in self.active if sid == service_id)


def arbitrate_ownership(events: list[LeaseEvent]) -> dict[str, dict[str, Any]]:
    """Pure replay of canonical events into final slot ownership.

    Re-sorting the input by ``event_seq`` makes the result independent of the
    physical order in which events were iterated. Each slot's final owner and
    every change basis are returned for ledger cross-checking.
    """
    ordered = sorted(events, key=lambda e: e.event_seq)
    owners: dict[tuple[str, int], dict[str, Any]] = {}
    history: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in ordered:
        key = (event.service_id, event.slot_index)
        if event.event_type == REASON_GRANT:
            owners[key] = {
                "lease_id": event.lease_id,
                "batch_id": event.batch_id,
                "quota_version": event.quota_version,
                "version": event.version,
                "last_event_seq": event.event_seq,
                "basis": event.basis,
            }
        elif event.event_type == REASON_RENEW:
            if key in owners:
                owners[key]["version"] = event.version
                owners[key]["last_event_seq"] = event.event_seq
                owners[key]["basis"] = event.basis
        else:  # expired / superseded
            owners.pop(key, None)
        history.setdefault(key, []).append({
            "event_seq": event.event_seq,
            "event_type": event.event_type,
            "lease_id": event.lease_id,
            "batch_id": event.batch_id,
            "version": event.version,
            "quota_version": event.quota_version,
            "basis": event.basis,
            "at_ms": event.at_ms,
        })
    result: dict[str, dict[str, Any]] = {}
    for (sid, slot), record in sorted(owners.items()):
        result[f"{sid}:slot{slot}"] = record
    return result

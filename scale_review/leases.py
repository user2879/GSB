from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, monotonic_ms
from .errors import LeaseError


@dataclass(frozen=True)
class Lease:
    lease_id: str
    service_id: str
    batch_id: str
    task_id: str
    replica_count: int
    global_slots: tuple[int, ...]
    service_slots: tuple[int, ...]
    granted_at_monotonic_ms: int
    expires_at_monotonic_ms: int
    version: int
    status: str
    settled_by_event_seq: int | None = None
    settlement_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "expires_at_monotonic_ms": self.expires_at_monotonic_ms,
            "global_slots": list(self.global_slots),
            "granted_at_monotonic_ms": self.granted_at_monotonic_ms,
            "lease_id": self.lease_id,
            "replica_count": self.replica_count,
            "service_id": self.service_id,
            "service_slots": list(self.service_slots),
            "settled_by_event_seq": self.settled_by_event_seq,
            "settlement_reason": self.settlement_reason,
            "status": self.status,
            "task_id": self.task_id,
            "version": self.version,
        }


class LeaseManager:
    """Owns replica quota slots and adjudicates terminal lease claims."""

    def __init__(
        self,
        global_replica_budget: int,
        service_concurrency_quotas: dict[str, int],
        lease_ttl_ms: int,
    ) -> None:
        self.global_replica_budget = global_replica_budget
        self.service_quotas = dict(service_concurrency_quotas)
        self.lease_ttl_ms = lease_ttl_ms
        self._lock = threading.RLock()
        self._global_owners: dict[int, str] = {}
        self._service_owners: dict[str, dict[int, str]] = {
            service_id: {} for service_id in sorted(service_concurrency_quotas)
        }
        self._leases: dict[str, Lease] = {}
        self._events: list[dict[str, Any]] = []
        self._event_seq = 0

    def _append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        self._event_seq += 1
        event["event_seq"] = self._event_seq
        event.setdefault("observed_monotonic_ms", monotonic_ms())
        self._events.append(event)
        return event

    def capacity_reason(self, service_id: str, replica_count: int) -> str | None:
        with self._lock:
            free_global = self.global_replica_budget - len(self._global_owners)
            if free_global < replica_count:
                return "global_replica_budget_exhausted"
            quota = self.service_quotas[service_id]
            owners = self._service_owners[service_id]
            if quota - len(owners) < replica_count:
                return "service_concurrency_quota_exhausted"
            return None

    def can_grant(self, service_id: str, replica_count: int = 1) -> bool:
        return self.capacity_reason(service_id, replica_count) is None

    def grant(
        self,
        lease_id: str,
        service_id: str,
        *,
        batch_id: str,
        task_id: str,
        replica_count: int = 1,
    ) -> Lease | None:
        if replica_count <= 0:
            raise ValueError("replica_count must be positive")
        with self._lock:
            if lease_id in self._leases:
                raise LeaseError(
                    "lease_id already exists",
                    field="lease_id",
                    details={"lease_id": lease_id},
                )
            reason = self.capacity_reason(service_id, replica_count)
            if reason is not None:
                return None

            global_slots = tuple(
                slot
                for slot in range(self.global_replica_budget)
                if slot not in self._global_owners
            )[:replica_count]
            service_limit = self.service_quotas[service_id]
            service_slots = tuple(
                slot
                for slot in range(service_limit)
                if slot not in self._service_owners[service_id]
            )[:replica_count]
            if len(global_slots) != replica_count or len(service_slots) != replica_count:
                return None

            now = monotonic_ms()
            lease = Lease(
                lease_id=lease_id,
                service_id=service_id,
                batch_id=batch_id,
                task_id=task_id,
                replica_count=replica_count,
                global_slots=global_slots,
                service_slots=service_slots,
                granted_at_monotonic_ms=now,
                expires_at_monotonic_ms=now + self.lease_ttl_ms,
                version=1,
                status="active",
            )
            for slot in global_slots:
                self._global_owners[slot] = lease_id
            service_owners = self._service_owners[service_id]
            for slot in service_slots:
                service_owners[slot] = lease_id
            self._leases[lease_id] = lease
            self._append_event(
                {
                    "event_type": "lease_granted",
                    "effective": True,
                    "lease_id": lease_id,
                    "service_id": service_id,
                    "batch_id": batch_id,
                    "lease_version": 1,
                    "replica_count": replica_count,
                    "global_slots": list(global_slots),
                    "service_slots": list(service_slots),
                    "basis": "lowest_free_slots_under_lock",
                }
            )
            return lease

    def get_lease(self, lease_id: str) -> Lease:
        with self._lock:
            if lease_id not in self._leases:
                raise LeaseError(
                    "lease not found",
                    field="lease_id",
                    details={"lease_id": lease_id, "reason": "not_found"},
                )
            return self._leases[lease_id]

    def renew(self, lease_id: str, expected_version: int) -> Lease:
        with self._lock:
            if lease_id not in self._leases:
                self._append_event(
                    {
                        "event_type": "renew_rejected",
                        "effective": False,
                        "lease_id": lease_id,
                        "expected_version": expected_version,
                        "basis": "lease_not_found",
                    }
                )
                raise LeaseError(
                    "lease not found",
                    field="lease_id",
                    details={"lease_id": lease_id, "reason": "not_found"},
                )

            lease = self._leases[lease_id]
            if lease.version != expected_version:
                event = self._append_event(
                    {
                        "event_type": "renew_rejected",
                        "effective": False,
                        "lease_id": lease_id,
                        "service_id": lease.service_id,
                        "lease_version": lease.version,
                        "expected_version": expected_version,
                        "basis": "optimistic_version_mismatch",
                    }
                )
                raise LeaseError(
                    "lease version mismatch; renewal rejected and expiry unchanged",
                    field="lease_version",
                    details={
                        "lease_id": lease_id,
                        "current_version": lease.version,
                        "expected_version": expected_version,
                        "event_seq": event["event_seq"],
                        "expires_at_monotonic_ms": lease.expires_at_monotonic_ms,
                        "reason": "version_conflict",
                    },
                )

            now = monotonic_ms()
            if lease.status != "active" or now >= lease.expires_at_monotonic_ms:
                if lease.status == "active":
                    self._settle_locked(lease_id, "expiry", now, "renew_after_deadline")
                self._append_event(
                    {
                        "event_type": "renew_rejected",
                        "effective": False,
                        "lease_id": lease_id,
                        "service_id": lease.service_id,
                        "lease_version": lease.version,
                        "expected_version": expected_version,
                        "basis": "lease_no_longer_active",
                    }
                )
                raise LeaseError(
                    "cannot renew expired or settled lease",
                    field="expires_at_monotonic_ms",
                    details={
                        "lease_id": lease_id,
                        "current_version": lease.version,
                        "status": self._leases[lease_id].status,
                        "reason": "lease_expired",
                    },
                )

            old_version = lease.version
            new_version = old_version + 1
            renewed = Lease(
                **{
                    **lease.to_dict(),
                    "granted_at_monotonic_ms": now,
                    "expires_at_monotonic_ms": now + self.lease_ttl_ms,
                    "version": new_version,
                    "status": "active",
                }
            )
            # to_dict keys match dataclass fields except no issue; construct explicitly.
            renewed = Lease(
                lease_id=lease.lease_id,
                service_id=lease.service_id,
                batch_id=lease.batch_id,
                task_id=lease.task_id,
                replica_count=lease.replica_count,
                global_slots=lease.global_slots,
                service_slots=lease.service_slots,
                granted_at_monotonic_ms=now,
                expires_at_monotonic_ms=now + self.lease_ttl_ms,
                version=new_version,
                status="active",
                settled_by_event_seq=None,
                settlement_reason=None,
            )
            self._leases[lease_id] = renewed
            self._append_event(
                {
                    "event_type": "lease_renewed",
                    "effective": True,
                    "lease_id": lease_id,
                    "service_id": lease.service_id,
                    "version_before": old_version,
                    "lease_version": new_version,
                    "expires_at_monotonic_ms": renewed.expires_at_monotonic_ms,
                    "basis": "optimistic_version_match",
                }
            )
            return renewed

    def _release_slots_locked(self, lease: Lease) -> None:
        for slot in lease.global_slots:
            if self._global_owners.get(slot) == lease.lease_id:
                del self._global_owners[slot]
        service_owners = self._service_owners[lease.service_id]
        for slot in lease.service_slots:
            if service_owners.get(slot) == lease.lease_id:
                del service_owners[slot]

    def _settle_locked(
        self,
        lease_id: str,
        source: str,
        now: int,
        basis: str,
    ) -> dict[str, Any]:
        lease = self._leases[lease_id]
        if lease.status != "active":
            return self._append_event(
                {
                    "event_type": "settlement_attempt",
                    "effective": False,
                    "source": source,
                    "lease_id": lease_id,
                    "service_id": lease.service_id,
                    "lease_version": lease.version,
                    "basis": basis,
                    "reason": f"already_{lease.status}",
                    "winning_event_seq": lease.settled_by_event_seq,
                }
            )

        if source == "expiry" and now < lease.expires_at_monotonic_ms:
            return self._append_event(
                {
                    "event_type": "expiry_check",
                    "effective": False,
                    "source": source,
                    "lease_id": lease_id,
                    "service_id": lease.service_id,
                    "lease_version": lease.version,
                    "basis": basis,
                    "reason": "not_expired",
                    "expires_at_monotonic_ms": lease.expires_at_monotonic_ms,
                    "observed_monotonic_ms": now,
                }
            )

        effective_source = source
        effective_basis = basis
        if source == "batch_revoke" and now >= lease.expires_at_monotonic_ms:
            self._append_event(
                {
                    "event_type": "settlement_attempt",
                    "effective": False,
                    "source": "batch_revoke",
                    "lease_id": lease_id,
                    "service_id": lease.service_id,
                    "lease_version": lease.version,
                    "basis": "expiry_deadline_has_objective_priority",
                    "reason": "expired",
                }
            )
            effective_source = "expiry"
            effective_basis = "expiry_wins_same_instant_conflict"

        status = "expired" if effective_source == "expiry" else "revoked"
        self._release_slots_locked(lease)
        released = Lease(
            lease_id=lease.lease_id,
            service_id=lease.service_id,
            batch_id=lease.batch_id,
            task_id=lease.task_id,
            replica_count=lease.replica_count,
            global_slots=lease.global_slots,
            service_slots=lease.service_slots,
            granted_at_monotonic_ms=lease.granted_at_monotonic_ms,
            expires_at_monotonic_ms=lease.expires_at_monotonic_ms,
            version=lease.version,
            status=status,
            settled_by_event_seq=self._event_seq + 1,
            settlement_reason=effective_source,
        )
        event = self._append_event(
            {
                "event_type": "quota_released",
                "effective": True,
                "source": effective_source,
                "requested_source": source,
                "lease_id": lease.lease_id,
                "service_id": lease.service_id,
                "batch_id": lease.batch_id,
                "lease_version": lease.version,
                "replica_count": lease.replica_count,
                "global_slots": list(lease.global_slots),
                "service_slots": list(lease.service_slots),
                "basis": effective_basis,
                "observed_monotonic_ms": now,
            }
        )
        settled = Lease(**{**released.to_dict(), "settled_by_event_seq": event["event_seq"]})
        self._leases[lease_id] = settled
        return event

    def reap_expired(self, service_id: str | None = None) -> list[dict[str, Any]]:
        now = monotonic_ms()
        with self._lock:
            lease_ids = sorted(self._leases)
            results: list[dict[str, Any]] = []
            for lease_id in lease_ids:
                lease = self._leases[lease_id]
                if service_id is not None and lease.service_id != service_id:
                    continue
                if lease.status == "active" and now >= lease.expires_at_monotonic_ms:
                    results.append(
                        self._settle_locked(lease_id, "expiry", now, "monotonic_deadline_reached")
                    )
            return results

    def revoke_batch(self, batch_id: str, service_id: str | None = None) -> list[dict[str, Any]]:
        now = monotonic_ms()
        with self._lock:
            lease_ids = sorted(
                lease_id
                for lease_id, lease in self._leases.items()
                if lease.batch_id == batch_id
                and (service_id is None or lease.service_id == service_id)
            )
            return [
                self._settle_locked(lease_id, "batch_revoke", now, "batch_undo_claim")
                for lease_id in lease_ids
            ]

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            service_state: dict[str, Any] = {}
            owner_set: list[list[Any]] = []
            for service_id in sorted(self.service_quotas):
                owners = self._service_owners[service_id]
                quota = self.service_quotas[service_id]
                service_state[service_id] = {
                    "quota": quota,
                    "held": len(owners),
                    "remaining": quota - len(owners),
                    "owners": {str(slot): lease_id for slot, lease_id in sorted(owners.items())},
                }
                for slot, lease_id in sorted(owners.items()):
                    lease = self._leases[lease_id]
                    owner_set.append([lease_id, service_id, slot, "service_slot"])
            global_owners = {
                str(slot): lease_id for slot, lease_id in sorted(self._global_owners.items())
            }
            for slot, lease_id in sorted(self._global_owners.items()):
                owner_set.append([lease_id, "__global__", slot, "global_slot"])
            return {
                "global_replica_budget": self.global_replica_budget,
                "global_held": len(self._global_owners),
                "global_remaining": self.global_replica_budget - len(self._global_owners),
                "global_owners": global_owners,
                "services": service_state,
                "owner_set": sorted(owner_set),
                "leases": [lease.to_dict() for lease_id, lease in sorted(self._leases.items())],
                "event_count": self._event_seq,
            }

    def snapshot_bytes(self) -> bytes:
        return canonical_bytes(self.snapshot())


def replay_quota_events(
    global_replica_budget: int,
    service_concurrency_quotas: dict[str, int],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay the ledger's canonical event sequence, never caller order."""

    manager = LeaseManager(
        global_replica_budget,
        service_concurrency_quotas,
        lease_ttl_ms=1,
    )
    # LeaseManager assigns its own sequence; replay directly against slot state
    # to preserve the original ledger event_seq in the returned proof.
    global_owners: dict[int, str] = {}
    service_owners = {service_id: {} for service_id in service_concurrency_quotas}
    applied: list[int] = []
    conflicts: list[dict[str, Any]] = []

    for event in sorted(events, key=lambda item: item["event_seq"]):
        if not event.get("effective"):
            continue
        lease_id = event["lease_id"]
        if event["event_type"] == "lease_granted":
            for slot in event["global_slots"]:
                if slot in global_owners and global_owners[slot] != lease_id:
                    conflicts.append(
                        {"event_seq": event["event_seq"], "slot": slot, "lease_id": lease_id}
                    )
                global_owners[slot] = lease_id
            for slot in event["service_slots"]:
                owners = service_owners[event["service_id"]]
                if slot in owners and owners[slot] != lease_id:
                    conflicts.append(
                        {"event_seq": event["event_seq"], "slot": slot, "lease_id": lease_id}
                    )
                owners[slot] = lease_id
            applied.append(event["event_seq"])
        elif event["event_type"] == "quota_released":
            for slot in event["global_slots"]:
                if global_owners.get(slot) == lease_id:
                    del global_owners[slot]
            for slot in event["service_slots"]:
                owners = service_owners[event["service_id"]]
                if owners.get(slot) == lease_id:
                    del owners[slot]
            applied.append(event["event_seq"])

    owner_set: list[list[Any]] = []
    for slot, lease_id in sorted(global_owners.items()):
        owner_set.append([lease_id, "__global__", slot, "global_slot"])
    for service_id in sorted(service_owners):
        for slot, lease_id in sorted(service_owners[service_id].items()):
            owner_set.append([lease_id, service_id, slot, "service_slot"])
    return {
        "global_remaining": global_replica_budget - len(global_owners),
        "global_held": len(global_owners),
        "service_remaining": {
            service_id: service_concurrency_quotas[service_id] - len(owners)
            for service_id, owners in sorted(service_owners.items())
        },
        "owner_set": sorted(owner_set),
        "applied_event_seq": applied,
        "conflicts": conflicts,
    }

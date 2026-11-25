from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .canonical import canonical_json, monotonic_ms, sha256_digest
from .errors import DiagnosticError, ErrorCode
from .protocol import WorkerClient
from .spec import DatabaseSpec


Clock = Callable[[], int]


@dataclass(slots=True)
class Lease:
    lease_id: str
    db_id: str
    shard_id: str
    event_seq: int
    granted_at_ms: int
    expires_at_ms: int
    renew_count: int
    lease_version: int
    route_version: int
    status: str = "active"
    terminal_at_ms: int | None = None
    terminal_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "db_id": self.db_id,
            "shard_id": self.shard_id,
            "event_seq": self.event_seq,
            "granted_at_ms": self.granted_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "renew_count": self.renew_count,
            "lease_version": self.lease_version,
            "route_version": self.route_version,
            "status": self.status,
            "terminal_at_ms": self.terminal_at_ms,
            "terminal_reason": self.terminal_reason,
        }


@dataclass(frozen=True, slots=True)
class LeaseTransition:
    lease_id: str
    db_id: str
    shard_id: str
    event_seq: int
    route_version: int
    from_status: str
    to_status: str
    reason: str
    basis: str
    at_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "db_id": self.db_id,
            "shard_id": self.shard_id,
            "event_seq": self.event_seq,
            "route_version": self.route_version,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "basis": self.basis,
            "at_ms": self.at_ms,
        }


@dataclass(frozen=True, slots=True)
class LeaseRequest:
    db_id: str
    shard_id: str
    event_seq: int
    route_version: int


class LeaseManager:
    """Single ownership authority for shard quota and lease generations."""

    def __init__(
        self,
        *,
        global_lease_limit: int,
        databases: Mapping[str, DatabaseSpec] | tuple[DatabaseSpec, ...],
        clock: Clock = monotonic_ms,
    ) -> None:
        if global_lease_limit <= 0:
            raise ValueError("global_lease_limit must be positive")
        db_specs = databases.values() if isinstance(databases, Mapping) else databases
        self.databases = {database.id: database for database in db_specs}
        for database in self.databases.values():
            if database.lease_limit <= 0:
                raise ValueError(f"database {database.id!r} lease_limit must be positive")
        self.global_lease_limit = global_lease_limit
        self.clock = clock
        self._lock = threading.RLock()
        self._leases: dict[str, Lease] = {}
        self._held_shards: dict[str, str] = {}
        self._active_by_db: dict[str, int] = {db_id: 0 for db_id in self.databases}
        self._active_global = 0
        self._transitions: list[LeaseTransition] = []

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _lease_id(self, db_id: str, shard_id: str, event_seq: int, route_version: int) -> str:
        material = canonical_json(
            {
                "db_id": db_id,
                "shard_id": shard_id,
                "event_seq": event_seq,
                "route_version": route_version,
            }
        )
        return "lease-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]

    def _validate_request(self, request: LeaseRequest) -> None:
        if request.db_id not in self.databases:
            raise DiagnosticError(ErrorCode.LEASE.value, f"unknown database {request.db_id!r}", field="db_id")
        if not request.shard_id:
            raise DiagnosticError(ErrorCode.LEASE.value, "shard_id must be non-empty", field="shard_id")
        if request.event_seq < 0:
            raise DiagnosticError(ErrorCode.LEASE.value, "event_seq must be non-negative", field="event_seq")
        if request.route_version <= 0:
            raise DiagnosticError(ErrorCode.LEASE.value, "route_version must be positive", field="route_version")

    def _check_acquire_locked(self, requests: list[LeaseRequest]) -> str | None:
        keys = [request.shard_id for request in requests]
        if len(keys) != len(set(keys)):
            return "duplicate shard request in the same event"
        for shard_id in keys:
            if shard_id in self._held_shards:
                holder = self._leases[self._held_shards[shard_id]]
                return (
                    f"shard {shard_id!r} on {holder.db_id!r} is already held by "
                    f"{self._held_shards[shard_id]} for event {holder.event_seq}"
                )
        if self._active_global + len(keys) > self.global_lease_limit:
            return "global lease limit would be exceeded"
        for database_id in sorted({request.db_id for request in requests}):
            count = sum(1 for request in requests if request.db_id == database_id)
            limit = self.databases[database_id].lease_limit
            if self._active_by_db[database_id] + count > limit:
                return f"database {database_id!r} lease limit {limit} would be exceeded"
        return None

    def can_acquire_many(self, requests: list[LeaseRequest]) -> tuple[bool, str]:
        with self._lock:
            reason = self._check_acquire_locked(requests)
            return (reason is None, reason or "available")

    def acquire_many(self, requests: list[LeaseRequest], *, ttl_ms: int) -> list[Lease]:
        if not requests:
            return []
        if ttl_ms <= 0:
            raise DiagnosticError(ErrorCode.LEASE.value, "lease ttl_ms must be positive", field="ttl_ms")
        with self._lock:
            for request in requests:
                self._validate_request(request)
            reason = self._check_acquire_locked(requests)
            if reason is not None:
                code = ErrorCode.LEASE.value if "shard" in reason or "duplicate" in reason else ErrorCode.QUOTA.value
                raise DiagnosticError(code, reason, field="lease")
            now = self.clock()
            created: list[Lease] = []
            for request in requests:
                lease = Lease(
                    lease_id=self._lease_id(
                        request.db_id,
                        request.shard_id,
                        request.event_seq,
                        request.route_version,
                    ),
                    db_id=request.db_id,
                    shard_id=request.shard_id,
                    event_seq=request.event_seq,
                    granted_at_ms=now,
                    expires_at_ms=now + ttl_ms,
                    renew_count=0,
                    lease_version=1,
                    route_version=request.route_version,
                )
                self._leases[lease.lease_id] = lease
                self._held_shards[lease.shard_id] = lease.lease_id
                self._active_by_db[lease.db_id] += 1
                self._active_global += 1
                self._transitions.append(
                    LeaseTransition(
                        lease_id=lease.lease_id,
                        db_id=lease.db_id,
                        shard_id=lease.shard_id,
                        event_seq=lease.event_seq,
                        route_version=lease.route_version,
                        from_status="unowned",
                        to_status="active",
                        reason="lease_granted",
                        basis="shard quota was unowned and all active lease limits allowed the grant",
                        at_ms=now,
                    )
                )
                created.append(lease)
            return created

    def acquire(
        self,
        request: LeaseRequest,
        *,
        ttl_ms: int,
    ) -> Lease:
        return self.acquire_many([request], ttl_ms=ttl_ms)[0]

    def get(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise DiagnosticError(ErrorCode.LEASE.value, f"lease {lease_id!r} is unknown", field="lease_id")
            return lease

    def _release_locked(
        self,
        lease: Lease,
        *,
        status: str,
        reason: str,
        basis: str,
        at_ms: int,
        record: bool,
    ) -> Lease:
        if lease.status != "active":
            return lease
        lease.status = status
        lease.terminal_at_ms = at_ms
        lease.terminal_reason = reason
        self._leases.pop(lease.lease_id, None)
        self._held_shards.pop(lease.shard_id, None)
        self._active_by_db[lease.db_id] -= 1
        self._active_global -= 1
        if record:
            self._transitions.append(
                LeaseTransition(
                    lease_id=lease.lease_id,
                    db_id=lease.db_id,
                    shard_id=lease.shard_id,
                    event_seq=lease.event_seq,
                    route_version=lease.route_version,
                    from_status="active",
                    to_status=status,
                    reason=reason,
                    basis=basis,
                    at_ms=at_ms,
                )
            )
        return lease

    def release(self, lease_id: str, *, reason: str = "probe_complete") -> Lease:
        with self._lock:
            lease = self.get(lease_id)
            return self._release_locked(
                lease,
                status="released",
                reason=reason,
                basis="probe finished normally",
                at_ms=self.clock(),
                record=True,
            )

    def release_events(self, event_seqs: set[int], *, reason: str = "uncommitted_batch_recovery") -> list[Lease]:
        with self._lock:
            now = self.clock()
            released: list[Lease] = []
            for lease in sorted(self._leases.values(), key=lambda item: item.lease_id):
                if lease.event_seq in event_seqs and lease.status == "active":
                    released.append(
                        self._release_locked(
                            lease,
                            status="reclaimed",
                            reason=reason,
                            basis="batch did not commit; quota returned before rerun",
                            at_ms=now,
                            record=True,
                        )
                    )
            return released

    def renew_local(self, lease_id: str, *, expected_lease_version: int, ttl_ms: int) -> Lease:
        if ttl_ms <= 0:
            raise DiagnosticError(ErrorCode.LEASE.value, "lease ttl_ms must be positive", field="ttl_ms")
        with self._lock:
            lease = self.get(lease_id)
            now = self.clock()
            if lease.status != "active" or now >= lease.expires_at_ms:
                if lease.status == "active":
                    self._release_locked(
                        lease,
                        status="expired",
                        reason="lease_expired",
                        basis="expires_at_ms <= local monotonic clock",
                        at_ms=now,
                        record=True,
                    )
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} is not active",
                    field="lease_id",
                )
            if expected_lease_version != lease.lease_version:
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} version {expected_lease_version} is stale; current version is {lease.lease_version}",
                    field="lease_version",
                    details={"expected": expected_lease_version, "actual": lease.lease_version},
                )
            previous_expiry = lease.expires_at_ms
            lease.lease_version += 1
            lease.renew_count += 1
            lease.expires_at_ms = now + ttl_ms
            self._transitions.append(
                LeaseTransition(
                    lease_id=lease.lease_id,
                    db_id=lease.db_id,
                    shard_id=lease.shard_id,
                    event_seq=lease.event_seq,
                    route_version=lease.route_version,
                    from_status="active",
                    to_status="renewed",
                    reason="lease_renewed",
                    basis=f"optimistic version {expected_lease_version} validated; expiry {previous_expiry}->{lease.expires_at_ms}",
                    at_ms=now,
                )
            )
            return lease

    def renew_with_worker(
        self,
        lease_id: str,
        *,
        expected_lease_version: int,
        ttl_ms: int,
        client: WorkerClient,
    ) -> Lease:
        """Validate the generation on the child before extending locally.

        A stale generation raises ``E_LEASE`` and leaves ``expires_at_ms``
        untouched.  The remote check is a real JSON-line request/response.
        """

        if ttl_ms <= 0:
            raise DiagnosticError(ErrorCode.LEASE.value, "lease ttl_ms must be positive", field="ttl_ms")
        with self._lock:
            lease = self.get(lease_id)
            now = self.clock()
            if lease.status != "active" or now >= lease.expires_at_ms:
                raise DiagnosticError(ErrorCode.LEASE.value, f"lease {lease_id!r} is not active", field="lease_id")
            binding = (lease.shard_id, lease.event_seq)

        remote = client.renew_lease(
            lease_id=lease_id,
            expected_lease_version=expected_lease_version,
            event_seq=binding[1],
            shard_id=binding[0],
        )
        remote_version = int(remote["lease_version"])

        with self._lock:
            lease = self._leases.get(lease_id)
            now = self.clock()
            if lease is None or lease.status != "active" or now >= lease.expires_at_ms:
                raise DiagnosticError(ErrorCode.LEASE.value, f"lease {lease_id!r} expired during renewal", field="lease_id")
            if lease.shard_id != binding[0] or lease.event_seq != binding[1]:
                raise DiagnosticError(ErrorCode.LEASE.value, "lease binding changed during renewal", field="lease_id")
            if expected_lease_version != lease.lease_version or remote_version != expected_lease_version + 1:
                raise DiagnosticError(
                    ErrorCode.LEASE.value,
                    f"lease {lease_id!r} version {expected_lease_version} is stale or changed concurrently; "
                    f"current version is {lease.lease_version}",
                    field="lease_version",
                    details={"expected": expected_lease_version, "actual": lease.lease_version, "remote": remote_version},
                )
            previous_expiry = lease.expires_at_ms
            lease.lease_version = remote_version
            lease.renew_count += 1
            lease.expires_at_ms = now + ttl_ms
            self._transitions.append(
                LeaseTransition(
                    lease_id=lease.lease_id,
                    db_id=lease.db_id,
                    shard_id=lease.shard_id,
                    event_seq=lease.event_seq,
                    route_version=lease.route_version,
                    from_status="active",
                    to_status="renewed",
                    reason="lease_renewed",
                    basis=f"worker validated optimistic version {expected_lease_version}; expiry {previous_expiry}->{lease.expires_at_ms}",
                    at_ms=now,
                )
            )
            return lease

    def expire_due(self, *, now_ms: int | None = None) -> list[Lease]:
        now = self.clock() if now_ms is None else now_ms
        with self._lock:
            expired: list[Lease] = []
            for lease in sorted(self._leases.values(), key=lambda item: item.lease_id):
                if lease.status == "active" and lease.expires_at_ms <= now:
                    expired.append(
                        self._release_locked(
                            lease,
                            status="expired",
                            reason="lease_expired",
                            basis="expires_at_ms <= local monotonic clock",
                            at_ms=now,
                            record=True,
                        )
                    )
            return expired

    def invalidate_generation(
        self,
        current_route_versions: Mapping[str, int],
        *,
        now_ms: int | None = None,
    ) -> list[Lease]:
        now = self.clock() if now_ms is None else now_ms
        with self._lock:
            superseded: list[Lease] = []
            for lease in sorted(self._leases.values(), key=lambda item: item.lease_id):
                current = current_route_versions.get(lease.db_id)
                if current is not None and lease.route_version < current:
                    superseded.append(
                        self._release_locked(
                            lease,
                            status="superseded",
                            reason="route_generation_superseded",
                            basis=f"lease route version {lease.route_version} < current {current}",
                            at_ms=now,
                            record=True,
                        )
                    )
            return superseded

    def reconcile(
        self,
        *,
        current_route_versions: Mapping[str, int] | None = None,
        now_ms: int | None = None,
    ) -> list[Lease]:
        """Canonically reconcile expiry and supersession in one pass.

        If one lease qualifies for both reasons, ``expired`` wins by a fixed
        precedence.  The set of leases is sorted by lease id, so calling this
        with the same logical state and time always yields the same ownership
        snapshot regardless of the caller's trigger ordering.
        """

        now = self.clock() if now_ms is None else now_ms
        versions = current_route_versions or {}
        with self._lock:
            reclaimed: list[Lease] = []
            for lease in sorted(self._leases.values(), key=lambda item: item.lease_id):
                expired = lease.status == "active" and lease.expires_at_ms <= now
                superseded_version = versions.get(lease.db_id)
                superseded = (
                    lease.status == "active"
                    and superseded_version is not None
                    and lease.route_version < superseded_version
                )
                if not expired and not superseded:
                    continue
                if expired:
                    status = "expired"
                    reason = "lease_expired"
                    basis = "expires_at_ms <= local monotonic clock; expiry precedence"
                else:
                    status = "superseded"
                    reason = "route_generation_superseded"
                    basis = (
                        f"lease route version {lease.route_version} < current {superseded_version}; "
                        "expiry not due"
                    )
                reclaimed.append(
                    self._release_locked(
                        lease,
                        status=status,
                        reason=reason,
                        basis=basis,
                        at_ms=now,
                        record=True,
                    )
                )
            return reclaimed

    def active_for_event(self, event_seq: int) -> list[Lease]:
        with self._lock:
            return sorted(
                (lease for lease in self._leases.values() if lease.event_seq == event_seq),
                key=lambda item: (item.db_id, item.shard_id, item.lease_id),
            )

    def earliest_expiry_ms(self) -> int | None:
        with self._lock:
            if not self._leases:
                return None
            return min(lease.expires_at_ms for lease in self._leases.values())

    def transitions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [transition.as_dict() for transition in self._transitions]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            held = [
                {
                    "db_id": lease.db_id,
                    "shard_id": lease.shard_id,
                    "lease_id": lease.lease_id,
                    "event_seq": lease.event_seq,
                    "route_version": lease.route_version,
                    "lease_version": lease.lease_version,
                }
                for lease in sorted(self._leases.values(), key=lambda item: (item.db_id, item.shard_id, item.lease_id))
            ]
            by_db = {
                db_id: {
                    "active_leases": self._active_by_db[db_id],
                    "lease_limit": self.databases[db_id].lease_limit,
                    "remaining_leases": self.databases[db_id].lease_limit - self._active_by_db[db_id],
                    "held_shards": sorted(
                        lease.shard_id
                        for lease in self._leases.values()
                        if lease.db_id == db_id
                    ),
                }
                for db_id in sorted(self.databases)
            }
            return {
                "global_lease_limit": self.global_lease_limit,
                "active_leases": self._active_global,
                "remaining_global": self.global_lease_limit - self._active_global,
                "by_db": by_db,
                "held_shards": held,
                "digest": sha256_digest(held),
            }

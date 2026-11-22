"""Real monotonic-clock entity leases with renewal, conflict and reclaim."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .clock import MonotonicClock
from .errors import (
    AppError,
    E_CONFIG_INVALID,
    E_LEASE_CONFLICT,
    E_LEASE_EXPIRED,
    E_LEASE_LOST,
    E_LEASE_WAIT_TIMEOUT,
)


def shard_for_entity(entity_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


@dataclass
class Lease:
    entity_id: str
    shard: int
    owner: str
    token: str
    version: str
    issued_ns: int
    expires_ns: int
    renewal_count: int = 0
    renewal_readings_ns: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "shard": self.shard,
            "owner": self.owner,
            "token": self.token,
            "version": self.version,
            "issued_ns": self.issued_ns,
            "expires_ns": self.expires_ns,
            "renewal_count": self.renewal_count,
            "renewal_readings_ns": list(self.renewal_readings_ns),
        }


class LeaseManager:
    def __init__(
        self,
        *,
        clock: MonotonicClock,
        ttl_ms: int,
        renew_interval_ms: int,
        shard_count: int,
        wait_timeout_ms: int,
    ) -> None:
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
            raise AppError(
                E_CONFIG_INVALID,
                "lease_ttl_ms must be positive",
                stage="lease",
                details={"lease_ttl_ms": ttl_ms},
            )
        if (
            isinstance(renew_interval_ms, bool)
            or not isinstance(renew_interval_ms, int)
            or renew_interval_ms <= 0
            or renew_interval_ms >= ttl_ms
        ):
            raise AppError(
                E_CONFIG_INVALID,
                "lease_renew_interval_ms must be positive and smaller than ttl",
                stage="lease",
                details={
                    "lease_renew_interval_ms": renew_interval_ms,
                    "lease_ttl_ms": ttl_ms,
                },
            )
        if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
            raise AppError(
                E_CONFIG_INVALID,
                "entity_shards must be positive",
                stage="lease",
                details={"entity_shards": shard_count},
            )
        self.clock = clock
        self.ttl_ms = ttl_ms
        self.renew_interval_ms = renew_interval_ms
        self.shard_count = shard_count
        self.wait_timeout_ms = wait_timeout_ms
        self._leases: dict[str, Lease] = {}
        self._condition = threading.Condition(threading.RLock())
        self.events: list[dict[str, Any]] = []

    def _record(self, kind: str, **payload: Any) -> None:
        event = {"event": kind, **payload}
        self.events.append(event)

    def _grant(
        self,
        entity_id: str,
        owner: str,
        version: str,
        *,
        now_ns: int,
        ttl_ms: int,
    ) -> Lease:
        lease = Lease(
            entity_id=entity_id,
            shard=shard_for_entity(entity_id, self.shard_count),
            owner=owner,
            token=uuid.uuid4().hex,
            version=version,
            issued_ns=now_ns,
            expires_ns=now_ns + ttl_ms * 1_000_000,
        )
        self._leases[entity_id] = lease
        self._record(
            "lease_granted",
            entity_id=entity_id,
            shard=lease.shard,
            owner=owner,
            token=lease.token,
            version=version,
            issued_ns=lease.issued_ns,
            expires_ns=lease.expires_ns,
            clock_source=self.clock.source,
        )
        return lease

    def _reclaim_locked(
        self,
        entity_id: str,
        existing: Lease,
        *,
        now_ns: int,
        reason: str,
        requester: str | None = None,
    ) -> None:
        del self._leases[entity_id]
        self._record(
            "lease_reclaimed",
            entity_id=entity_id,
            shard=existing.shard,
            previous_owner=existing.owner,
            previous_token=existing.token,
            previous_version=existing.version,
            reclaimed_at_ns=now_ns,
            clock_source=self.clock.source,
            basis="monotonic_deadline_reached",
            reason=reason,
            expires_ns=existing.expires_ns,
            requester=requester,
        )
        self._condition.notify_all()

    def acquire(
        self,
        entity_id: str,
        *,
        owner: str,
        version: str,
        ttl_ms: int | None = None,
        wait_timeout_ms: int | None = None,
    ) -> Lease:
        effective_ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        if (
            isinstance(effective_ttl, bool)
            or not isinstance(effective_ttl, int)
            or effective_ttl <= 0
        ):
            raise AppError(
                E_CONFIG_INVALID,
                "injected/runtime lease ttl must be positive",
                stage="lease",
                entity_id=entity_id,
                details={"lease_ttl_ms": effective_ttl},
            )
        timeout_ms = self.wait_timeout_ms if wait_timeout_ms is None else wait_timeout_ms
        wait_deadline_ns = self.clock.now_ns() + max(1, timeout_ms) * 1_000_000
        conflict_reported = False
        while True:
            with self._condition:
                now_ns = self.clock.now_ns()
                existing = self._leases.get(entity_id)
                if existing is None:
                    return self._grant(
                        entity_id,
                        owner,
                        version,
                        now_ns=now_ns,
                        ttl_ms=effective_ttl,
                    )
                if now_ns >= existing.expires_ns:
                    self._reclaim_locked(
                        entity_id,
                        existing,
                        now_ns=now_ns,
                        reason="expired_before_conflict_acquire",
                        requester=owner,
                    )
                    return self._grant(
                        entity_id,
                        owner,
                        version,
                        now_ns=now_ns,
                        ttl_ms=effective_ttl,
                    )
                if not conflict_reported:
                    conflict_reported = True
                    self._record(
                        "lease_conflict",
                        code=E_LEASE_CONFLICT,
                        entity_id=entity_id,
                        shard=existing.shard,
                        requester=owner,
                        holder=existing.owner,
                        clock_reading_ns=now_ns,
                        expires_ns=existing.expires_ns,
                        action="wait_for_reclaim",
                        clock_source=self.clock.source,
                    )
                remaining_wait_ns = wait_deadline_ns - now_ns
                if remaining_wait_ns <= 0:
                    raise AppError(
                        E_LEASE_WAIT_TIMEOUT,
                        "timed out waiting for entity lease reclaim",
                        stage="lease",
                        entity_id=entity_id,
                        details={
                            "holder": existing.owner,
                            "clock_reading_ns": now_ns,
                            "expires_ns": existing.expires_ns,
                        },
                    )
                until_expiry_ns = max(0, existing.expires_ns - now_ns)
                wait_ns = min(remaining_wait_ns, until_expiry_ns + 1_000_000)
                self._condition.wait(wait_ns / 1_000_000_000)

    def renew(
        self,
        entity_id: str,
        token: str,
        *,
        ttl_ms: int | None = None,
    ) -> Lease:
        effective_ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        if (
            isinstance(effective_ttl, bool)
            or not isinstance(effective_ttl, int)
            or effective_ttl <= 0
        ):
            raise AppError(
                E_CONFIG_INVALID,
                "renewal ttl must be positive",
                stage="lease",
                entity_id=entity_id,
                details={"lease_ttl_ms": effective_ttl},
            )
        with self._condition:
            now_ns = self.clock.now_ns()
            existing = self._leases.get(entity_id)
            if existing is None or existing.token != token:
                raise AppError(
                    E_LEASE_LOST,
                    "lease token is no longer active",
                    stage="lease",
                    entity_id=entity_id,
                    details={"clock_reading_ns": now_ns},
                )
            if now_ns >= existing.expires_ns:
                self._reclaim_locked(
                    entity_id,
                    existing,
                    now_ns=now_ns,
                    reason="renewal_after_expiry_rejected",
                    requester=existing.owner,
                )
                self._record(
                    "lease_expired",
                    code=E_LEASE_EXPIRED,
                    entity_id=entity_id,
                    owner=existing.owner,
                    token=existing.token,
                    clock_reading_ns=now_ns,
                    expires_ns=existing.expires_ns,
                    clock_source=self.clock.source,
                )
                raise AppError(
                    E_LEASE_EXPIRED,
                    "renewal attempted at or after expiry; entity was released",
                    stage="lease",
                    entity_id=entity_id,
                    details={
                        "clock_reading_ns": now_ns,
                        "expires_ns": existing.expires_ns,
                    },
                )
            existing.expires_ns = now_ns + effective_ttl * 1_000_000
            existing.renewal_count += 1
            existing.renewal_readings_ns.append(now_ns)
            self._record(
                "lease_renewed",
                entity_id=entity_id,
                owner=existing.owner,
                token=existing.token,
                clock_reading_ns=now_ns,
                new_expires_ns=existing.expires_ns,
                renewal_count=existing.renewal_count,
                clock_source=self.clock.source,
            )
            self._condition.notify_all()
            return existing

    def validate(self, entity_id: str, token: str) -> Lease:
        with self._condition:
            now_ns = self.clock.now_ns()
            existing = self._leases.get(entity_id)
            if existing is None or existing.token != token:
                raise AppError(
                    E_LEASE_LOST,
                    "lease is not held by this work unit",
                    stage="lease",
                    entity_id=entity_id,
                )
            if now_ns >= existing.expires_ns:
                self._reclaim_locked(
                    entity_id,
                    existing,
                    now_ns=now_ns,
                    reason="validation_after_expiry",
                    requester=existing.owner,
                )
                self._record(
                    "lease_expired",
                    code=E_LEASE_EXPIRED,
                    entity_id=entity_id,
                    owner=existing.owner,
                    token=existing.token,
                    clock_reading_ns=now_ns,
                    expires_ns=existing.expires_ns,
                    clock_source=self.clock.source,
                )
                raise AppError(
                    E_LEASE_EXPIRED,
                    "lease expired before commit",
                    stage="lease",
                    entity_id=entity_id,
                    details={
                        "clock_reading_ns": now_ns,
                        "expires_ns": existing.expires_ns,
                    },
                )
            return existing

    def release(self, entity_id: str, token: str) -> None:
        with self._condition:
            existing = self._leases.get(entity_id)
            if existing is None or existing.token != token:
                return
            del self._leases[entity_id]
            self._record(
                "lease_released",
                entity_id=entity_id,
                owner=existing.owner,
                token=existing.token,
                clock_reading_ns=self.clock.now_ns(),
                clock_source=self.clock.source,
            )
            self._condition.notify_all()

    def reclaim_expired(self) -> list[dict[str, Any]]:
        reclaimed: list[dict[str, Any]] = []
        with self._condition:
            now_ns = self.clock.now_ns()
            for entity_id, existing in list(self._leases.items()):
                if now_ns >= existing.expires_ns:
                    before = len(self.events)
                    self._reclaim_locked(
                        entity_id,
                        existing,
                        now_ns=now_ns,
                        reason="reaper_detected_expiry",
                    )
                    reclaimed.extend(self.events[before:])
        return reclaimed

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "leases": {key: self._leases[key].to_dict() for key in sorted(self._leases)},
                "events": list(self.events),
            }

    def conflict_failures(self) -> list[dict[str, Any]]:
        from .errors import make_failure

        return [
            make_failure(
                E_LEASE_CONFLICT,
                "entity is held by another work unit; request waited for reclaim",
                stage="lease",
                entity_id=event.get("entity_id"),
                details={
                    "requester": event.get("requester"),
                    "holder": event.get("holder"),
                    "clock_reading_ns": event.get("clock_reading_ns"),
                    "expires_ns": event.get("expires_ns"),
                    "action": "wait_for_reclaim",
                },
            )
            for event in self.events
            if event.get("event") == "lease_conflict"
        ]


class LeaseGuard:
    """Hold a lease and renew it in the background while a worker computes."""

    def __init__(
        self,
        manager: LeaseManager,
        *,
        entity_id: str,
        owner: str,
        version: str,
        ttl_ms: int | None = None,
    ) -> None:
        self.manager = manager
        self.entity_id = entity_id
        self.owner = owner
        self.version = version
        self.ttl_ms = ttl_ms
        self.lease: Lease | None = None
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LeaseGuard":
        self.lease = self.manager.acquire(
            self.entity_id,
            owner=self.owner,
            version=self.version,
            ttl_ms=self.ttl_ms,
        )
        self._thread = threading.Thread(
            target=self._renew_loop,
            name=f"lease-renew-{self.entity_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _renew_loop(self) -> None:
        assert self.lease is not None
        while not self._stop.wait(self.manager.renew_interval_ms / 1000):
            try:
                self.manager.renew(self.entity_id, self.lease.token, ttl_ms=self.ttl_ms)
            except BaseException as exc:  # noqa: BLE001 - propagated on exit
                self.error = exc
                self._stop.set()
                return

    def validate(self) -> None:
        if self.error is not None:
            raise self.error
        if self.lease is None:
            raise AppError(E_LEASE_LOST, "lease guard is not active", stage="lease")
        self.manager.validate(self.entity_id, self.lease.token)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.lease is not None:
            self.manager.release(self.entity_id, self.lease.token)
        if exc is None and self.error is not None:
            raise self.error

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from .errors import LeaseError
from .events import monotonic_ms
from .models import LeasePolicy


class LeaseSink(Protocol):
    def lease_changed(self, lease: "Lease", status: str) -> None: ...


@dataclass
class Lease:
    object_key: str
    shard_index: int
    granted_ms: int
    expires_ms: int
    version: int
    renew_count: int = 0
    status: str = "active"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class LeaseManager:
    """In-memory lease authority backed by the real monotonic clock.

    Active leases alone consume quota.  Expiration is evaluated lazily on every
    grant/renew/expire/snapshot operation, and the procedure actually releases
    both global and per-object counters.
    """

    def __init__(
        self,
        *,
        global_quota: int,
        clock_ms: Callable[[], int] = monotonic_ms,
        sink: LeaseSink | None = None,
        event_callback: Callable[[dict[str, object]], None] | None = None,
        initial_version: int = 0,
    ) -> None:
        if global_quota <= 0:
            raise ValueError("global_quota must be positive")
        self.global_quota = global_quota
        self.clock_ms = clock_ms
        self.sink = sink
        self.event_callback = event_callback
        self._lock = threading.RLock()
        self._active: dict[tuple[str, int], Lease] = {}
        self._per_object: Counter[str] = Counter()
        self._version = int(initial_version)
        self.grant_count = 0
        self.renew_count = 0
        self.renew_rejected_count = 0
        self.duplicate_rejected_count = 0
        self.expiration_count = 0
        self.reclaim_count = 0
        self.peak_global = 0
        self.peak_per_object: dict[str, int] = {}

    @property
    def active_count(self) -> int:
        with self._lock:
            self.expire_due_locked(self.clock_ms())
            return len(self._active)

    def _notify(self, event: str, **fields: object) -> None:
        if self.event_callback is not None:
            payload: dict[str, object] = {"event": event, "monotonic_ms": self.clock_ms()}
            payload.update(fields)
            self.event_callback(payload)

    def _sink(self, lease: Lease, status: str) -> None:
        if self.sink is not None:
            self.sink.lease_changed(lease, status)

    def _remove_active_locked(self, lease: Lease) -> bool:
        key = (lease.object_key, lease.shard_index)
        current = self._active.get(key)
        if current is None or current.version != lease.version:
            return False
        del self._active[key]
        self._per_object[lease.object_key] -= 1
        if self._per_object[lease.object_key] <= 0:
            del self._per_object[lease.object_key]
        return True

    def expire_due_locked(self, now_ms: int) -> list[Lease]:
        expired: list[Lease] = []
        for key, lease in list(self._active.items()):
            if now_ms >= lease.expires_ms:
                lease.status = "expired"
                self._remove_active_locked(lease)
                self.expiration_count += 1
                expired.append(lease)
                self._sink(lease, "expired")
                self._notify(
                    "lease_expired",
                    object_key=lease.object_key,
                    shard_index=lease.shard_index,
                    version=lease.version,
                    expired_ms=lease.expires_ms,
                )
        return expired

    def expire_due(self) -> list[Lease]:
        with self._lock:
            return self.expire_due_locked(self.clock_ms())

    def active_object_count(self, object_key: str) -> int:
        with self._lock:
            self.expire_due_locked(self.clock_ms())
            return self._per_object.get(object_key, 0)

    def can_grant(self, object_key: str, policy: LeasePolicy) -> bool:
        with self._lock:
            self.expire_due_locked(self.clock_ms())
            return (
                len(self._active) < self.global_quota
                and self._per_object.get(object_key, 0) < policy.max_concurrent_leases
            )

    def grant(self, object_key: str, shard_index: int, policy: LeasePolicy) -> Lease:
        with self._lock:
            now = self.clock_ms()
            self.expire_due_locked(now)
            existing = self._active.get((object_key, shard_index))
            if existing is not None:
                self.duplicate_rejected_count += 1
                self._notify(
                    "duplicate_lease_rejected",
                    object_key=object_key,
                    shard_index=shard_index,
                    version=existing.version,
                )
                return Lease(**existing.as_dict())
            if len(self._active) >= self.global_quota:
                raise LeaseError(
                    "全局租约配额已满",
                    object_key=object_key,
                    shard_index=shard_index,
                    detail={
                        "limit": self.global_quota,
                        "active": len(self._active),
                        "scope": "global",
                    },
                )
            object_active = self._per_object.get(object_key, 0)
            if object_active >= policy.max_concurrent_leases:
                raise LeaseError(
                    "对象级租约配额已满",
                    object_key=object_key,
                    shard_index=shard_index,
                    detail={
                        "limit": policy.max_concurrent_leases,
                        "active": object_active,
                        "scope": "object",
                    },
                )
            self._version += 1
            lease = Lease(
                object_key=object_key,
                shard_index=shard_index,
                granted_ms=now,
                expires_ms=now + policy.lease_ms,
                version=self._version,
                renew_count=0,
                status="active",
            )
            self._active[(object_key, shard_index)] = lease
            self._per_object[object_key] += 1
            self.grant_count += 1
            self.peak_global = max(self.peak_global, len(self._active))
            self.peak_per_object[object_key] = max(
                self.peak_per_object.get(object_key, 0), self._per_object[object_key]
            )
            self._sink(lease, "active")
            self._notify(
                "lease_granted",
                object_key=object_key,
                shard_index=shard_index,
                version=lease.version,
                expires_ms=lease.expires_ms,
            )
            return Lease(**lease.as_dict())

    def renew(self, lease: Lease, duration_ms: int) -> Lease:
        with self._lock:
            now = self.clock_ms()
            self.expire_due_locked(now)
            current = self._active.get((lease.object_key, lease.shard_index))
            if current is None or current.version != lease.version:
                self.renew_rejected_count += 1
                self._notify(
                    "lease_renew_rejected",
                    object_key=lease.object_key,
                    shard_index=lease.shard_index,
                    version=lease.version,
                )
                raise LeaseError(
                    "租约版本不匹配或租约已失效",
                    object_key=lease.object_key,
                    shard_index=lease.shard_index,
                    detail={
                        "presented_version": lease.version,
                        "active_version": None if current is None else current.version,
                    },
                )
            if now >= current.expires_ms:
                self.renew_rejected_count += 1
                raise LeaseError(
                    "租约已到期，不能延长",
                    object_key=lease.object_key,
                    shard_index=lease.shard_index,
                    detail={"expires_ms": current.expires_ms, "now_ms": now},
                )
            current.expires_ms = now + duration_ms
            current.renew_count += 1
            self.renew_count += 1
            self._sink(current, "active")
            self._notify(
                "lease_renewed",
                object_key=current.object_key,
                shard_index=current.shard_index,
                version=current.version,
                renew_count=current.renew_count,
                expires_ms=current.expires_ms,
            )
            return Lease(**current.as_dict())

    def assert_valid(self, lease: Lease) -> None:
        """Validate without extending; expiry first reclaims the quota."""

        with self._lock:
            now = self.clock_ms()
            self.expire_due_locked(now)
            current = self._active.get((lease.object_key, lease.shard_index))
            if current is None or current.version != lease.version:
                raise LeaseError(
                    "租约已失效，不能提交批次",
                    object_key=lease.object_key,
                    shard_index=lease.shard_index,
                    detail={"presented_version": lease.version},
                )
            if now >= current.expires_ms:
                raise LeaseError(
                    "租约已到期，不能提交批次",
                    object_key=lease.object_key,
                    shard_index=lease.shard_index,
                    detail={"expires_ms": current.expires_ms, "now_ms": now},
                )

    def release(self, lease: Lease, status: str = "released") -> bool:
        with self._lock:
            current = self._active.get((lease.object_key, lease.shard_index))
            if current is None or current.version != lease.version:
                return False
            self._remove_active_locked(current)
            current.status = status
            self._sink(current, status)
            self._notify(
                "lease_released",
                object_key=current.object_key,
                shard_index=current.shard_index,
                version=current.version,
                status=status,
            )
            return True

    def reclaim_all(self, status: str = "reclaimed") -> list[Lease]:
        with self._lock:
            reclaimed: list[Lease] = []
            for lease in list(self._active.values()):
                lease.status = status
                self._remove_active_locked(lease)
                self.reclaim_count += 1
                self._sink(lease, status)
                reclaimed.append(Lease(**lease.as_dict()))
            return reclaimed

    def snapshot(self, *, expire: bool = True) -> dict[str, object]:
        with self._lock:
            now = self.clock_ms()
            if expire:
                self.expire_due_locked(now)
            return {
                "now_monotonic_ms": now,
                "global_limit": self.global_quota,
                "active_global": len(self._active),
                "remaining_global": self.global_quota - len(self._active),
                "per_object_active": dict(sorted(self._per_object.items())),
                "peak_global": self.peak_global,
                "peak_per_object": dict(sorted(self.peak_per_object.items())),
                "grant_count": self.grant_count,
                "renew_count": self.renew_count,
                "duplicate_rejected_count": self.duplicate_rejected_count,
                "renew_rejected_count": self.renew_rejected_count,
                "expiration_count": self.expiration_count,
                "reclaim_count": self.reclaim_count,
                "active_leases": [
                    lease.as_dict()
                    for lease in sorted(self._active.values(), key=lambda item: (item.object_key, item.shard_index))
                ],
            }

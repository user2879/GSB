"""Concurrent lease issuance, optimistic renewal, and real expiry.

The manager owns all three admission constraints:

* a global active-lease limit,
* a per-subscriber active-lease limit, and
* a per-channel sliding one-second dispatch-rate limit.

Admission, rate-window timestamps, counters, wait accounting, and expiry are
all mutated while holding the same condition lock.  A caller can therefore
never observe or create an over-budget lease.
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Callable

from .canonical import MonotonicClock
from .errors import LeaseError
from .logging_utils import StructuredLogger
from .models import Acquisition, Lease, QuotaSnapshot, WaitRecord


@dataclass(frozen=True)
class LeaseExpiry:
    revoked_mono_ms: float
    lease_id: str
    batch_id: str
    subscriber_id: str
    channel_id: str
    version: int
    expires_mono_ms: float


@dataclass(frozen=True)
class LeaseSnapshot:
    lease_id: str
    batch_id: str
    subscriber_id: str
    channel_id: str
    granted_mono_ms: float
    expires_mono_ms: float
    version: int


class LeaseManager:
    def __init__(
        self,
        *,
        global_limit: int,
        subscriber_limits: dict[str, int],
        channel_rate_limits: dict[str, int],
        clock: MonotonicClock | None = None,
        rate_window_ms: int = 1000,
        logger: StructuredLogger | None = None,
        lease_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if global_limit <= 0:
            raise ValueError("global_limit must be positive")
        if rate_window_ms <= 0:
            raise ValueError("rate_window_ms must be positive")
        self.global_limit = int(global_limit)
        self.subscriber_limits = {key: int(value) for key, value in subscriber_limits.items()}
        self.channel_rate_limits = {
            key: int(value) for key, value in channel_rate_limits.items()
        }
        self.clock = clock or MonotonicClock()
        self.rate_window_ms = int(rate_window_ms)
        self.logger = logger
        self._lease_id_factory = lease_id_factory or (
            lambda: "lease_" + uuid.uuid4().hex
        )

        self._condition = threading.Condition(threading.RLock())
        self._active: dict[str, Lease] = {}
        self._subscriber_active: dict[str, int] = defaultdict(int)
        self._channel_sends: dict[str, deque[float]] = defaultdict(deque)
        self._granted_versions: dict[str, int] = {}
        self._expiries: list[LeaseExpiry] = []
        self._revoked: list[LeaseExpiry] = []

    @property
    def active_count(self) -> int:
        with self._condition:
            return len(self._active)

    def active_for_subscriber(self, subscriber_id: str) -> int:
        with self._condition:
            return self._subscriber_active.get(subscriber_id, 0)

    def channel_window_count(self, channel_id: str, *, now_mono_ms: float | None = None) -> int:
        with self._condition:
            now = self.clock.milliseconds() if now_mono_ms is None else now_mono_ms
            self._prune_rate_locked(channel_id, now)
            return len(self._channel_sends[channel_id])

    def acquire(
        self,
        *,
        batch_id: str,
        subscriber_id: str,
        channel_id: str,
        ttl_ms: int,
        wait_start_mono_ms: float | None = None,
    ) -> Acquisition:
        self._validate_request(
            batch_id=batch_id,
            subscriber_id=subscriber_id,
            channel_id=channel_id,
            ttl_ms=ttl_ms,
        )
        wait_start = (
            self.clock.milliseconds()
            if wait_start_mono_ms is None
            else float(wait_start_mono_ms)
        )
        waited_reason = "none"
        with self._condition:
            while True:
                now = self.clock.milliseconds()
                self._expire_locked(now)
                self._prune_rate_locked(channel_id, now)

                reason = self._blocking_reason_locked(now, subscriber_id, channel_id)
                if reason == "none":
                    return self._grant_locked(
                        now_mono_ms=now,
                        batch_id=batch_id,
                        subscriber_id=subscriber_id,
                        channel_id=channel_id,
                        ttl_ms=ttl_ms,
                        wait_start_mono_ms=wait_start,
                        wait_reason=waited_reason,
                    )

                if waited_reason == "none":
                    waited_reason = reason
                timeout = self._wait_timeout_seconds_locked(now, subscriber_id, channel_id)
                self._condition.wait(timeout=timeout)

    def try_acquire(
        self,
        *,
        batch_id: str,
        subscriber_id: str,
        channel_id: str,
        ttl_ms: int,
        wait_start_mono_ms: float | None = None,
    ) -> tuple[Acquisition | None, str]:
        """Attempt admission without blocking.

        The scheduler uses this to keep quota-blocked batches out of worker
        threads, so one slow channel cannot cause head-of-line blocking for a
        different channel with available capacity.
        """

        self._validate_request(
            batch_id=batch_id,
            subscriber_id=subscriber_id,
            channel_id=channel_id,
            ttl_ms=ttl_ms,
        )
        wait_start = (
            self.clock.milliseconds()
            if wait_start_mono_ms is None
            else float(wait_start_mono_ms)
        )
        with self._condition:
            now = self.clock.milliseconds()
            self._expire_locked(now)
            self._prune_rate_locked(channel_id, now)
            reason = self._blocking_reason_locked(now, subscriber_id, channel_id)
            if reason != "none":
                return None, reason
            acquisition = self._grant_locked(
                now_mono_ms=now,
                batch_id=batch_id,
                subscriber_id=subscriber_id,
                channel_id=channel_id,
                ttl_ms=ttl_ms,
                wait_start_mono_ms=wait_start,
                wait_reason="none",
            )
            return acquisition, "none"

    def wait_for_change(self, timeout_seconds: float) -> None:
        with self._condition:
            self._condition.wait(timeout=max(0.001, float(timeout_seconds)))

    def _validate_request(
        self,
        *,
        batch_id: str,
        subscriber_id: str,
        channel_id: str,
        ttl_ms: int,
    ) -> None:
        if ttl_ms <= 0:
            raise LeaseError(
                "lease ttl must be positive", batch_id=batch_id, subscriber_id=subscriber_id
            )
        if subscriber_id not in self.subscriber_limits:
            raise LeaseError(
                f"unknown subscriber {subscriber_id!r}",
                batch_id=batch_id,
                subscriber_id=subscriber_id,
            )
        if channel_id not in self.channel_rate_limits:
            raise LeaseError(
                f"unknown channel {channel_id!r}",
                batch_id=batch_id,
                subscriber_id=subscriber_id,
            )

    def _grant_locked(
        self,
        *,
        now_mono_ms: float,
        batch_id: str,
        subscriber_id: str,
        channel_id: str,
        ttl_ms: int,
        wait_start_mono_ms: float,
        wait_reason: str,
    ) -> Acquisition:
        lease = Lease(
            lease_id=self._lease_id_factory(),
            batch_id=batch_id,
            subscriber_id=subscriber_id,
            channel_id=channel_id,
            granted_mono_ms=now_mono_ms,
            expires_mono_ms=now_mono_ms + float(ttl_ms),
            version=1,
        )
        self._active[lease.lease_id] = lease
        self._granted_versions[lease.lease_id] = lease.version
        self._subscriber_active[subscriber_id] += 1
        self._channel_sends[channel_id].append(now_mono_ms)
        quota = QuotaSnapshot(
            global_limit=self.global_limit,
            subscriber_limit=self.subscriber_limits[subscriber_id],
            channel_rate_limit=self.channel_rate_limits[channel_id],
            global_active_after_grant=len(self._active),
            subscriber_active_after_grant=self._subscriber_active[subscriber_id],
            channel_sends_in_window=len(self._channel_sends[channel_id]),
        )
        wait = WaitRecord(
            start_mono_ms=wait_start_mono_ms,
            end_mono_ms=now_mono_ms,
            wait_ms=max(0.0, now_mono_ms - wait_start_mono_ms),
            reason=wait_reason,
        )
        if self.logger:
            self.logger.info(
                "E_LEASE_GRANTED",
                "lease granted",
                lease_id=lease.lease_id,
                batch_id=batch_id,
                subscriber_id=subscriber_id,
                channel_id=channel_id,
                lease_version=lease.version,
                wait_ms=round(wait.wait_ms, 3),
            )
        return Acquisition(lease=lease, wait=wait, quota=quota)

    def renew(
        self,
        lease_id: str,
        *,
        subscriber_id: str,
        expected_version: int,
        extension_ms: int,
    ) -> Lease:
        if extension_ms <= 0:
            raise LeaseError(
                "renewal extension_ms must be positive",
                lease_id=lease_id,
                subscriber_id=subscriber_id,
            )
        with self._condition:
            now = self.clock.milliseconds()
            self._expire_locked(now)
            lease = self._active.get(lease_id)
            if lease is None:
                error = LeaseError(
                    "lease is no longer active; renewal rejected",
                    lease_id=lease_id,
                    subscriber_id=subscriber_id,
                    details={
                        "expected_version": int(expected_version),
                    },
                )
                if self.logger:
                    self.logger.diagnostic(error)
                raise error
            if lease.subscriber_id != subscriber_id:
                error = LeaseError(
                    "lease is held by a different subscriber; renewal rejected",
                    lease_id=lease_id,
                    batch_id=lease.batch_id,
                    subscriber_id=subscriber_id,
                    details={
                        "holder_subscriber_id": lease.subscriber_id,
                        "expected_version": int(expected_version),
                    },
                )
                if self.logger:
                    self.logger.diagnostic(error)
                raise error
            if lease.version != expected_version:
                error = LeaseError(
                    "lease version mismatch; renewal rejected without extension",
                    lease_id=lease_id,
                    batch_id=lease.batch_id,
                    subscriber_id=subscriber_id,
                    details={
                        "expected_version": int(expected_version),
                        "current_version": lease.version,
                    },
                )
                if self.logger:
                    self.logger.diagnostic(error)
                raise error

            renewed = replace(
                lease,
                version=lease.version + 1,
                expires_mono_ms=now + float(extension_ms),
            )
            self._active[lease_id] = renewed
            self._granted_versions[lease_id] = renewed.version
            if self.logger:
                self.logger.info(
                    "E_LEASE_RENEWED",
                    "lease renewed",
                    lease_id=lease_id,
                    batch_id=renewed.batch_id,
                    subscriber_id=renewed.subscriber_id,
                    lease_version=renewed.version,
                    expires_mono_ms=renewed.expires_mono_ms,
                )
            return renewed

    def validate(self, lease_id: str, expected_version: int | None = None) -> bool:
        with self._condition:
            now = self.clock.milliseconds()
            self._expire_locked(now)
            lease = self._active.get(lease_id)
            if lease is None:
                return False
            if expected_version is not None and lease.version != expected_version:
                return False
            return True

    def release(self, lease_id: str, expected_version: int | None = None) -> bool:
        with self._condition:
            lease = self._active.get(lease_id)
            if lease is None:
                return False
            if expected_version is not None and lease.version != expected_version:
                return False
            self._active.pop(lease_id, None)
            self._subscriber_active[lease.subscriber_id] = max(
                0, self._subscriber_active[lease.subscriber_id] - 1
            )
            self._condition.notify_all()
            if self.logger:
                self.logger.info(
                    "E_LEASE_RELEASED",
                    "lease released",
                    lease_id=lease.lease_id,
                    batch_id=lease.batch_id,
                    subscriber_id=lease.subscriber_id,
                    lease_version=lease.version,
                )
            return True

    def expire(self, now_mono_ms: float | None = None) -> list[LeaseExpiry]:
        with self._condition:
            now = self.clock.milliseconds() if now_mono_ms is None else float(now_mono_ms)
            expired = self._expire_locked(now)
            for event in expired:
                if self.logger:
                    self.logger.error(
                        "E_LEASE_EXPIRED",
                        "lease expired and was reclaimed",
                        lease_id=event.lease_id,
                        batch_id=event.batch_id,
                        subscriber_id=event.subscriber_id,
                        channel_id=event.channel_id,
                        lease_version=event.version,
                        revoked_mono_ms=event.revoked_mono_ms,
                        expires_mono_ms=event.expires_mono_ms,
                    )
            return expired

    def _expire_locked(self, now_mono_ms: float) -> list[LeaseExpiry]:
        expired_ids = [
            lease_id
            for lease_id, lease in self._active.items()
            if lease.expires_mono_ms <= now_mono_ms
        ]
        expired_ids.sort()
        expired: list[LeaseExpiry] = []
        for lease_id in expired_ids:
            lease = self._active.pop(lease_id)
            self._subscriber_active[lease.subscriber_id] = max(
                0, self._subscriber_active[lease.subscriber_id] - 1
            )
            event = LeaseExpiry(
                revoked_mono_ms=now_mono_ms,
                lease_id=lease.lease_id,
                batch_id=lease.batch_id,
                subscriber_id=lease.subscriber_id,
                channel_id=lease.channel_id,
                version=lease.version,
                expires_mono_ms=lease.expires_mono_ms,
            )
            self._expiries.append(event)
            expired.append(event)
        if expired:
            self._condition.notify_all()
        return expired

    def revoke_all(self, reason: str = "declaration_replaced") -> list[LeaseExpiry]:
        with self._condition:
            now = self.clock.milliseconds()
            lease_ids = sorted(self._active)
            revoked: list[LeaseExpiry] = []
            for lease_id in lease_ids:
                lease = self._active.pop(lease_id)
                self._subscriber_active[lease.subscriber_id] = max(
                    0, self._subscriber_active[lease.subscriber_id] - 1
                )
                event = LeaseExpiry(
                    revoked_mono_ms=now,
                    lease_id=lease.lease_id,
                    batch_id=lease.batch_id,
                    subscriber_id=lease.subscriber_id,
                    channel_id=lease.channel_id,
                    version=lease.version,
                    expires_mono_ms=lease.expires_mono_ms,
                )
                self._revoked.append(event)
                revoked.append(event)
                if self.logger:
                    self.logger.warning(
                        "E_LEASE_REVOKED",
                        reason,
                        lease_id=lease.lease_id,
                        batch_id=lease.batch_id,
                        subscriber_id=lease.subscriber_id,
                        lease_version=lease.version,
                        revoked_mono_ms=now,
                    )
            self._condition.notify_all()
            return revoked

    def snapshot(self) -> list[LeaseSnapshot]:
        with self._condition:
            now = self.clock.milliseconds()
            self._expire_locked(now)
            return [
                LeaseSnapshot(
                    lease_id=lease.lease_id,
                    batch_id=lease.batch_id,
                    subscriber_id=lease.subscriber_id,
                    channel_id=lease.channel_id,
                    granted_mono_ms=lease.granted_mono_ms,
                    expires_mono_ms=lease.expires_mono_ms,
                    version=lease.version,
                )
                for lease in sorted(self._active.values(), key=lambda item: item.lease_id)
            ]

    def expiry_history(self) -> list[LeaseExpiry]:
        with self._condition:
            return list(self._expiries)

    def revoke_history(self) -> list[LeaseExpiry]:
        with self._condition:
            return list(self._revoked)

    def _blocking_reason_locked(
        self, now_mono_ms: float, subscriber_id: str, channel_id: str
    ) -> str:
        if len(self._active) >= self.global_limit:
            return "global_lease_limit"
        if self._subscriber_active[subscriber_id] >= self.subscriber_limits[subscriber_id]:
            return "subscriber_lease_limit"
        timestamps = self._channel_sends[channel_id]
        if len(timestamps) >= self.channel_rate_limits[channel_id]:
            return "channel_rate_limit"
        return "none"

    def _wait_timeout_seconds_locked(
        self, now_mono_ms: float, subscriber_id: str, channel_id: str
    ) -> float:
        candidates: list[float] = []
        if len(self._active) >= self.global_limit:
            candidates.extend(
                max(0.0, lease.expires_mono_ms - now_mono_ms)
                for lease in self._active.values()
            )
        if self._subscriber_active[subscriber_id] >= self.subscriber_limits[subscriber_id]:
            candidates.extend(
                max(0.0, lease.expires_mono_ms - now_mono_ms)
                for lease in self._active.values()
                if lease.subscriber_id == subscriber_id
            )
        timestamps = self._channel_sends[channel_id]
        if len(timestamps) >= self.channel_rate_limits[channel_id]:
            candidates.append(
                max(0.0, timestamps[0] + self.rate_window_ms - now_mono_ms)
            )
        positive = [value for value in candidates if value > 0.0]
        seconds = (min(positive) / 1000.0) if positive else 0.01
        return min(max(seconds, 0.001), 0.05)

    def _prune_rate_locked(self, channel_id: str, now_mono_ms: float) -> None:
        timestamps = self._channel_sends[channel_id]
        boundary = now_mono_ms - self.rate_window_ms
        while timestamps and timestamps[0] <= boundary:
            timestamps.popleft()

    def sorted_active_batch_ids(self) -> list[str]:
        with self._condition:
            return sorted(lease.batch_id for lease in self._active.values())

    def limits_snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "global_limit": self.global_limit,
                "global_active": len(self._active),
                "subscriber_limits": dict(sorted(self.subscriber_limits.items())),
                "subscriber_active": {
                    key: value
                    for key, value in sorted(self._subscriber_active.items())
                    if value
                },
                "channel_rate_limits": dict(sorted(self.channel_rate_limits.items())),
            }

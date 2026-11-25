from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .canonical import monotonic_ms
from .errors import LeaseError
from .models import Batch


@dataclass
class Lease:
    lease_id: str
    batch_id: str
    worker_id: str
    version: int
    granted_monotonic_ms: int
    expires_monotonic_ms: int
    ttl_ms: int
    status: str = "active"
    release_reason: str | None = None
    released_monotonic_ms: int | None = None
    renewals: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueueEntry:
    batch_id: str
    worker_id: str
    batch_index: int
    wait_started_monotonic_ms: int
    wait_ended_monotonic_ms: int | None = None

    @property
    def wait_ms(self) -> int | None:
        if self.wait_ended_monotonic_ms is None:
            return None
        return self.wait_ended_monotonic_ms - self.wait_started_monotonic_ms

    def to_json(self, now_ms: int | None = None) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_index": self.batch_index,
            "wait_started_monotonic_ms": self.wait_started_monotonic_ms,
            "wait_ended_monotonic_ms": self.wait_ended_monotonic_ms,
            "wait_ms": self.wait_ms,
            "current_wait_ms": None if self.wait_ended_monotonic_ms is not None else (now_ms or monotonic_ms()) - self.wait_started_monotonic_ms,
        }


ProcessFn = Callable[[Batch, Lease], dict[str, Any]]


class LeaseScheduler:
    """Schedule batches under global and per-worker concurrency leases."""

    def __init__(
        self,
        worker_quotas: dict[str, int],
        global_concurrency: int,
        lease_ttl_ms: int = 5_000,
        clock: Callable[[], int] = monotonic_ms,
    ) -> None:
        if global_concurrency <= 0:
            raise ValueError("global_concurrency must be positive")
        self.worker_quotas = dict(sorted(worker_quotas.items()))
        self.global_concurrency = int(global_concurrency)
        self.lease_ttl_ms = int(lease_ttl_ms)
        self._clock = clock
        self._condition = threading.Condition()
        self._pending: list[Batch] = []
        self._waiting: dict[str, QueueEntry] = {}
        self._active: dict[str, Lease] = {}
        self._running_global = 0
        self._running_worker: dict[str, int] = {worker_id: 0 for worker_id in self.worker_quotas}
        self.events: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, BaseException] = {}

    def _now(self) -> int:
        return int(self._clock())

    def _event(self, code: str, **details: Any) -> None:
        event = {"monotonic_ms": self._now(), "code": code, **details}
        self.events.append(event)

    def _stable_pending(self) -> list[Batch]:
        return sorted(self._pending, key=lambda batch: (batch.worker_id, batch.batch_index, batch.batch_id))

    def enqueue(self, batches: list[Batch] | tuple[Batch, ...]) -> None:
        with self._condition:
            for batch in sorted(batches, key=lambda item: (item.worker_id, item.batch_index, item.batch_id)):
                if batch.batch_id in self._waiting or batch.batch_id in self._active:
                    continue
                self._pending.append(batch)
                self._waiting[batch.batch_id] = QueueEntry(
                    batch_id=batch.batch_id,
                    worker_id=batch.worker_id,
                    batch_index=batch.batch_index,
                    wait_started_monotonic_ms=self._now(),
                )
                self._event("queued", batch_id=batch.batch_id, worker_id=batch.worker_id)
            self._condition.notify_all()

    def _can_dispatch_locked(self, batch: Batch) -> bool:
        quota = self.worker_quotas.get(batch.worker_id, 1)
        return self._running_global < self.global_concurrency and self._running_worker.get(batch.worker_id, 0) < quota

    def _grant_locked(self, batch: Batch) -> Lease:
        now = self._now()
        entry = self._waiting.pop(batch.batch_id, None)
        lease = Lease(
            lease_id=f"lease_{batch.batch_id}_{uuid.uuid4().hex[:8]}",
            batch_id=batch.batch_id,
            worker_id=batch.worker_id,
            version=1,
            granted_monotonic_ms=now,
            expires_monotonic_ms=now + self.lease_ttl_ms,
            ttl_ms=self.lease_ttl_ms,
        )
        if entry is not None:
            entry.wait_ended_monotonic_ms = now
            lease.wait_started_monotonic_ms = entry.wait_started_monotonic_ms  # type: ignore[attr-defined]
            lease.wait_ended_monotonic_ms = now  # type: ignore[attr-defined]
            lease.wait_ms = entry.wait_ms  # type: ignore[attr-defined]
            self._event(
                "wait_finished",
                batch_id=batch.batch_id,
                wait_started_monotonic_ms=entry.wait_started_monotonic_ms,
                wait_ended_monotonic_ms=now,
                wait_ms=entry.wait_ms,
            )
        self._running_global += 1
        self._running_worker[batch.worker_id] = self._running_worker.get(batch.worker_id, 0) + 1
        self._active[lease.lease_id] = lease
        self._event(
            "granted",
            lease_id=lease.lease_id,
            batch_id=batch.batch_id,
            worker_id=batch.worker_id,
            version=lease.version,
            granted_monotonic_ms=lease.granted_monotonic_ms,
            expires_monotonic_ms=lease.expires_monotonic_ms,
        )
        return lease

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int | None = None) -> Lease:
        with self._condition:
            lease = self._active.get(lease_id)
            if lease is None or lease.status != "active":
                current_version = lease.version if lease else None
                raise LeaseError(
                    "租约不存在或已终结，版本乐观校验失败",
                    lease_id=lease_id,
                    expected_version=expected_version,
                    current_version=current_version,
                )
            if lease.version != expected_version:
                raise LeaseError(
                    "租约版本不符，拒绝续约且不延长到期时刻",
                    lease_id=lease_id,
                    expected_version=expected_version,
                    current_version=lease.version,
                    expires_monotonic_ms=lease.expires_monotonic_ms,
                )
            now = self._now()
            if now >= lease.expires_monotonic_ms:
                self._release_once_locked(lease, "lease_expired")
                raise LeaseError(
                    "租约已到期，不能续约",
                    lease_id=lease_id,
                    version=lease.version,
                    expires_monotonic_ms=lease.expires_monotonic_ms,
                    now_monotonic_ms=now,
                )
            lease.version += 1
            lease.renewals += 1
            lease.ttl_ms = int(ttl_ms or lease.ttl_ms)
            lease.expires_monotonic_ms = now + lease.ttl_ms
            self._event(
                "renewed",
                lease_id=lease_id,
                version=lease.version,
                expires_monotonic_ms=lease.expires_monotonic_ms,
            )
            self._condition.notify_all()
            return lease

    def complete(self, lease_id: str, expected_version: int) -> Lease:
        with self._condition:
            lease = self._active.get(lease_id)
            if lease is None:
                raise LeaseError("完成回执引用了未知租约", lease_id=lease_id, expected_version=expected_version)
            if lease.version != expected_version or lease.status != "active":
                raise LeaseError(
                    "完成确认的租约版本或状态不符",
                    lease_id=lease_id,
                    expected_version=expected_version,
                    current_version=lease.version,
                    status=lease.status,
                    release_reason=lease.release_reason,
                )
            self._release_once_locked(lease, "completed")
            return lease

    def _release_once_locked(self, lease: Lease, reason: str) -> bool:
        if lease.release_reason is not None or lease.status != "active":
            self._event(
                "release_skipped",
                lease_id=lease.lease_id,
                batch_id=lease.batch_id,
                requested_reason=reason,
                accepted_reason=lease.release_reason,
            )
            return False
        now = self._now()
        lease.status = "released" if reason == "completed" else reason
        lease.release_reason = reason
        lease.released_monotonic_ms = now
        self._running_global = max(0, self._running_global - 1)
        self._running_worker[lease.worker_id] = max(0, self._running_worker.get(lease.worker_id, 0) - 1)
        self._event(
            "released",
            lease_id=lease.lease_id,
            batch_id=lease.batch_id,
            worker_id=lease.worker_id,
            reason=reason,
            released_monotonic_ms=now,
            running_global=self._running_global,
            running_worker=self._running_worker[lease.worker_id],
        )
        self._condition.notify_all()
        return True

    def expire_due(self) -> list[Lease]:
        expired: list[Lease] = []
        with self._condition:
            now = self._now()
            for lease in list(self._active.values()):
                if lease.status == "active" and now >= lease.expires_monotonic_ms:
                    if self._release_once_locked(lease, "lease_expired"):
                        expired.append(lease)
        return expired

    def reclaim_quota(self, worker_id: str, new_quota: int) -> list[Lease]:
        if new_quota <= 0:
            raise ValueError("new_quota must be positive")
        reclaimed: list[Lease] = []
        with self._condition:
            self.worker_quotas[worker_id] = int(new_quota)
            excess = self._running_worker.get(worker_id, 0) - int(new_quota)
            if excess <= 0:
                self._condition.notify_all()
                return reclaimed
            candidates = sorted(
                (lease for lease in self._active.values() if lease.worker_id == worker_id and lease.status == "active"),
                key=lambda lease: (lease.granted_monotonic_ms, lease.batch_id),
                reverse=True,
            )
            for lease in candidates:
                if excess <= 0:
                    break
                if self._release_once_locked(lease, "quota_reclaimed"):
                    reclaimed.append(lease)
                    excess -= 1
            self._condition.notify_all()
        return reclaimed

    def water_snapshot(self) -> dict[str, Any]:
        with self._condition:
            workers = {}
            for worker_id, quota in sorted(self.worker_quotas.items()):
                running = self._running_worker.get(worker_id, 0)
                workers[worker_id] = {
                    "running": running,
                    "quota": quota,
                    "water_level": f"{running}/{quota}",
                    "utilization": 0.0 if quota == 0 else running / quota,
                }
            return {
                "global": {
                    "running": self._running_global,
                    "limit": self.global_concurrency,
                    "water_level": f"{self._running_global}/{self.global_concurrency}",
                },
                "workers": workers,
                "active_leases": [lease.to_json() for lease in sorted(self._active.values(), key=lambda item: item.batch_id)],
            }

    def queue_snapshot(self) -> list[dict[str, Any]]:
        now = self._now()
        with self._condition:
            entries = list(self._waiting.values())
        return [entry.to_json(now) for entry in sorted(entries, key=lambda item: (item.worker_id, item.batch_index, item.batch_id))]

    def run(
        self,
        batches: list[Batch] | tuple[Batch, ...],
        process_fn: ProcessFn,
        *,
        on_grant: Callable[[Batch, Lease], None] | None = None,
        on_finish: Callable[[Batch, Lease, dict[str, Any]], None] | None = None,
        poll_ms: int = 5,
    ) -> dict[str, dict[str, Any]]:
        self.enqueue(list(batches))
        threads: list[threading.Thread] = []

        def run_one(batch: Batch, lease: Lease) -> None:
            try:
                result = process_fn(batch, lease)
                completed = self.complete(lease.lease_id, lease.version)
                if on_finish is not None:
                    on_finish(batch, completed, result)
                self.results[batch.batch_id] = result
            except BaseException as exc:  # captured for deterministic verification
                try:
                    if lease.status == "active":
                        self.complete(lease.lease_id, lease.version)
                except LeaseError as completion_error:
                    self.errors[batch.batch_id] = completion_error
                else:
                    self.errors[batch.batch_id] = exc
                if isinstance(exc, Exception):
                    self._event("failed", batch_id=batch.batch_id, error_type=type(exc).__name__, message=str(exc))

        while True:
            with self._condition:
                self.expire_due()
                dispatchable: Batch | None = None
                for batch in self._stable_pending():
                    if self._can_dispatch_locked(batch):
                        dispatchable = batch
                        break
                if dispatchable is not None:
                    self._pending.remove(dispatchable)
                    lease = self._grant_locked(dispatchable)
                    if on_grant is not None:
                        try:
                            on_grant(dispatchable, lease)
                        except Exception as exc:
                            self._release_once_locked(lease, "grant_callback_failed")
                            self.errors[dispatchable.batch_id] = exc
                            continue
                    thread = threading.Thread(
                        target=run_one,
                        args=(dispatchable, lease),
                        name=f"batch-{dispatchable.batch_id[:12]}",
                        daemon=True,
                    )
                    threads.append(thread)
                    thread.start()
                    continue
                if not self._pending and self._running_global == 0:
                    break
                timeout = max(0.001, poll_ms / 1000.0)
                if self._active:
                    next_expiry = min(lease.expires_monotonic_ms for lease in self._active.values() if lease.status == "active")
                    timeout = min(timeout, max(0.001, (next_expiry - self._now()) / 1000.0))
                self._condition.wait(timeout=timeout)

        for thread in threads:
            thread.join(timeout=10)
        return dict(sorted(self.results.items()))

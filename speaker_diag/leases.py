"""Batch lease scheduling with three independent quotas and real expiry.

A grant consumes one global lease, one per-worker lease and one per-worker
concurrency permit.  Expiry is decided from ``time.monotonic_ns``; every
decision is appended to an event log that can be replayed without wall-clock
data.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .canonical import canonical_sha256, floor_ratio
from .errors import DiagnosticError


@dataclass(frozen=True)
class Lease:
    lease_id: str
    batch_id: str
    worker_id: str
    granted_ns: int
    expires_ns: int
    version: int
    queue_start_ns: int
    queue_end_ns: int

    @property
    def granted_ms(self) -> int:
        return self.granted_ns // 1_000_000

    @property
    def expires_ms(self) -> int:
        return self.expires_ns // 1_000_000

    @property
    def queue_ms(self) -> int:
        return (self.queue_end_ns - self.queue_start_ns) // 1_000_000

    def to_dict(self, origin_ns: int | None = None) -> dict[str, Any]:
        result = {
            "lease_id": self.lease_id,
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "granted_monotonic_ms": self.granted_ms,
            "expires_monotonic_ms": self.expires_ms,
            "version": self.version,
            "queue_start_monotonic_ms": self.queue_start_ns // 1_000_000,
            "queue_end_monotonic_ms": self.queue_end_ns // 1_000_000,
            "queue_duration_ms": self.queue_ms,
        }
        if origin_ns is not None:
            result["granted_relative_ms"] = (self.granted_ns - origin_ns) // 1_000_000
            result["expires_relative_ms"] = (self.expires_ns - origin_ns) // 1_000_000
            result["queue_start_relative_ms"] = (self.queue_start_ns - origin_ns) // 1_000_000
            result["queue_end_relative_ms"] = (self.queue_end_ns - origin_ns) // 1_000_000
        return result


class LeaseScheduler:
    def __init__(
        self,
        *,
        global_limit: int,
        worker_lease_limits: Mapping[str, int],
        worker_concurrency_limits: Mapping[str, int],
        lease_ttl_ms: Mapping[str, int],
        metric_scale: int = 10_000,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if global_limit < 1:
            raise DiagnosticError("E_LEASE", "global lease limit must be positive", field="global_batch_leases")
        self.global_limit = int(global_limit)
        self.worker_lease_limits = {worker: int(limit) for worker, limit in worker_lease_limits.items()}
        self.worker_concurrency_limits = {
            worker: int(limit) for worker, limit in worker_concurrency_limits.items()
        }
        self.lease_ttl_ms = {worker: int(ttl) for worker, ttl in lease_ttl_ms.items()}
        if any(limit < 1 for limit in self.worker_lease_limits.values()):
            raise DiagnosticError("E_LEASE", "worker lease limits must be positive", field="max_worker_leases")
        if any(limit < 1 for limit in self.worker_concurrency_limits.values()):
            raise DiagnosticError("E_LEASE", "worker concurrency limits must be positive", field="max_concurrent")
        if any(ttl < 1 for ttl in self.lease_ttl_ms.values()):
            raise DiagnosticError("E_LEASE", "lease TTL must be positive", field="lease_ms")
        self.metric_scale = int(metric_scale)
        self.clock_ns = clock_ns
        self.origin_ns = self.clock_ns()
        self.condition = threading.Condition(threading.RLock())
        self.active: dict[str, Lease] = {}
        self.active_per_worker: Counter[str] = Counter()
        self.concurrent_per_worker: Counter[str] = Counter()
        self.expired_counts: Counter[str] = Counter()
        self.queue_records: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.lease_sequence = 0
        self.event_sequence = 0

    @classmethod
    def from_spec(cls, spec: Any) -> "LeaseScheduler":
        return cls(
            global_limit=spec.limits.global_batch_leases,
            worker_lease_limits={worker.worker_id: worker.max_worker_leases for worker in spec.workers},
            worker_concurrency_limits={worker.worker_id: worker.max_concurrent for worker in spec.workers},
            lease_ttl_ms={worker.worker_id: worker.lease_ms for worker in spec.workers},
            metric_scale=spec.limits.metric_scale,
        )

    def _append_event(self, event_type: str, *, now_ns: int, lease: Lease | None = None, **fields: Any) -> None:
        self.event_sequence += 1
        event = {
            "seq": self.event_sequence,
            "type": event_type,
            "monotonic_ms": now_ns // 1_000_000,
            "relative_ms": (now_ns - self.origin_ns) // 1_000_000,
            **fields,
        }
        if lease is not None:
            event.update(
                {
                    "lease_id": lease.lease_id,
                    "batch_id": lease.batch_id,
                    "worker_id": lease.worker_id,
                    "version": lease.version,
                    "granted_monotonic_ms": lease.granted_ms,
                    "expires_monotonic_ms": lease.expires_ms,
                    "queue_start_monotonic_ms": lease.queue_start_ns // 1_000_000,
                    "queue_end_monotonic_ms": lease.queue_end_ns // 1_000_000,
                }
            )
        self.events.append(event)

    def _expire_locked(self, now_ns: int) -> None:
        expired = [lease for lease in self.active.values() if lease.expires_ns <= now_ns]
        for lease in sorted(expired, key=lambda item: (item.expires_ns, item.lease_id)):
            self.active.pop(lease.lease_id, None)
            self.active_per_worker[lease.worker_id] -= 1
            self.concurrent_per_worker[lease.worker_id] -= 1
            self.expired_counts[lease.worker_id] += 1
            self._append_event("EXPIRED", now_ns=now_ns, lease=lease, effective_expiry_ms=lease.expires_ms)
        if expired:
            self.condition.notify_all()

    def _can_acquire_locked(self, worker_id: str) -> bool:
        return (
            len(self.active) < self.global_limit
            and self.active_per_worker[worker_id] < self.worker_lease_limits.get(worker_id, 0)
            and self.concurrent_per_worker[worker_id] < self.worker_concurrency_limits.get(worker_id, 0)
        )

    def _next_expiry_ns(self) -> int | None:
        if not self.active:
            return None
        return min(lease.expires_ns for lease in self.active.values())

    def acquire(self, batch_id: str, worker_id: str, *, timeout_ms: int | None = None) -> Lease:
        if worker_id not in self.worker_lease_limits:
            raise DiagnosticError("E_LEASE", "unknown worker in lease scheduler", field="worker_id", details={"worker_id": worker_id})
        if worker_id not in self.lease_ttl_ms:
            raise DiagnosticError("E_LEASE", "missing lease TTL for worker", field="lease_ms", details={"worker_id": worker_id})
        queue_start = self.clock_ns()
        deadline = None if timeout_ms is None else queue_start + int(timeout_ms) * 1_000_000
        with self.condition:
            self._append_event(
                "QUEUED",
                now_ns=queue_start,
                batch_id=batch_id,
                worker_id=worker_id,
                timeout_ms=timeout_ms,
            )
            while True:
                now = self.clock_ns()
                self._expire_locked(now)
                if self._can_acquire_locked(worker_id):
                    self.lease_sequence += 1
                    queue_end = now
                    ttl_ns = self.lease_ttl_ms[worker_id] * 1_000_000
                    lease = Lease(
                        lease_id=f"L{self.lease_sequence:08d}",
                        batch_id=batch_id,
                        worker_id=worker_id,
                        granted_ns=now,
                        expires_ns=now + ttl_ns,
                        version=1,
                        queue_start_ns=queue_start,
                        queue_end_ns=queue_end,
                    )
                    self.active[lease.lease_id] = lease
                    self.active_per_worker[worker_id] += 1
                    self.concurrent_per_worker[worker_id] += 1
                    self.queue_records.append(
                        {
                            "batch_id": batch_id,
                            "worker_id": worker_id,
                            "queue_start_monotonic_ms": queue_start // 1_000_000,
                            "queue_end_monotonic_ms": queue_end // 1_000_000,
                            "queue_duration_ms": (queue_end - queue_start) // 1_000_000,
                            "lease_id": lease.lease_id,
                        }
                    )
                    self._append_event("GRANTED", now_ns=now, lease=lease)
                    return lease
                if deadline is not None and now >= deadline:
                    self._append_event(
                        "QUEUE_TIMEOUT",
                        now_ns=now,
                        batch_id=batch_id,
                        worker_id=worker_id,
                        timeout_ms=timeout_ms,
                    )
                    raise DiagnosticError(
                        "E_LEASE",
                        "timed out waiting for lease and quota availability",
                        field="quota",
                        details={"batch_id": batch_id, "worker_id": worker_id, "timeout_ms": timeout_ms},
                    )
                wake_at = self._next_expiry_ns()
                if deadline is not None and (wake_at is None or deadline < wake_at):
                    wake_at = deadline
                if wake_at is None:
                    wait_seconds = None
                else:
                    wait_seconds = max(0.001, (wake_at - now) / 1_000_000_000.0)
                self.condition.wait(timeout=wait_seconds)

    def renew(self, lease_id: str, expected_version: int, *, ttl_ms: int | None = None) -> Lease:
        now = self.clock_ns()
        with self.condition:
            self._expire_locked(now)
            lease = self.active.get(lease_id)
            if lease is None:
                self._append_event(
                    "RENEW_REJECTED",
                    now_ns=now,
                    lease_id=lease_id,
                    reason="not_active_or_expired",
                )
                raise DiagnosticError(
                    "E_LEASE",
                    "lease is not active or has already expired",
                    field="lease_id",
                    details={"lease_id": lease_id},
                )
            if expected_version != lease.version:
                self._append_event(
                    "RENEW_REJECTED",
                    now_ns=now,
                    lease=lease,
                    reason="version_mismatch",
                    expected_version=expected_version,
                )
                raise DiagnosticError(
                    "E_LEASE",
                    "optimistic lease version check failed",
                    field="version",
                    details={"lease_id": lease_id, "expected": expected_version, "actual": lease.version},
                )
            extension_ms = self.lease_ttl_ms[lease.worker_id] if ttl_ms is None else int(ttl_ms)
            if extension_ms < 1:
                raise DiagnosticError("E_LEASE", "renewal TTL must be positive", field="ttl_ms")
            renewed = Lease(
                lease_id=lease.lease_id,
                batch_id=lease.batch_id,
                worker_id=lease.worker_id,
                granted_ns=lease.granted_ns,
                expires_ns=now + extension_ms * 1_000_000,
                version=lease.version + 1,
                queue_start_ns=lease.queue_start_ns,
                queue_end_ns=lease.queue_end_ns,
            )
            self.active[lease_id] = renewed
            self._append_event("RENEWED", now_ns=now, lease=renewed, ttl_ms=extension_ms)
            return renewed

    def release(self, lease_id: str, *, expected_version: int | None = None) -> bool:
        now = self.clock_ns()
        with self.condition:
            self._expire_locked(now)
            lease = self.active.get(lease_id)
            if lease is None:
                return False
            if expected_version is not None and expected_version != lease.version:
                raise DiagnosticError(
                    "E_LEASE",
                    "optimistic release version check failed",
                    field="version",
                    details={"lease_id": lease_id, "expected": expected_version, "actual": lease.version},
                )
            self.active.pop(lease_id, None)
            self.active_per_worker[lease.worker_id] -= 1
            self.concurrent_per_worker[lease.worker_id] -= 1
            self._append_event("RELEASED", now_ns=now, lease=lease)
            self.condition.notify_all()
            return True

    def snapshot(self) -> dict[str, Any]:
        now = self.clock_ns()
        with self.condition:
            self._expire_locked(now)
            workers: list[dict[str, Any]] = []
            for worker_id in sorted(self.worker_lease_limits):
                lease_limit = self.worker_lease_limits[worker_id]
                concurrency_limit = self.worker_concurrency_limits.get(worker_id, 0)
                active_leases = int(self.active_per_worker[worker_id])
                active_concurrency = int(self.concurrent_per_worker[worker_id])
                workers.append(
                    {
                        "worker_id": worker_id,
                        "lease_limit": lease_limit,
                        "concurrency_limit": concurrency_limit,
                        "active_leases": active_leases,
                        "active_concurrency": active_concurrency,
                        "lease_watermark_q": floor_ratio(active_leases, lease_limit, self.metric_scale),
                        "concurrency_watermark_q": floor_ratio(
                            active_concurrency, concurrency_limit, self.metric_scale
                        ),
                        "expired_leases": int(self.expired_counts[worker_id]),
                    }
                )
            return {
                "schema": "speaker-diag-lease-snapshot-v1",
                "clock": "time.monotonic_ns",
                "origin_monotonic_ms": self.origin_ns // 1_000_000,
                "now_monotonic_ms": now // 1_000_000,
                "metric_scale": self.metric_scale,
                "truncation": "floor_non_negative",
                "global_limit": self.global_limit,
                "global_active_leases": len(self.active),
                "global_watermark_q": floor_ratio(len(self.active), self.global_limit, self.metric_scale),
                "workers": workers,
                "active_leases": [
                    self.active[key].to_dict(self.origin_ns) for key in sorted(self.active)
                ],
                "queue_records": list(self.queue_records),
                "event_count": len(self.events),
                "event_digest": canonical_sha256(self.events),
            }

    def replay(self) -> dict[str, Any]:
        with self.condition:
            return replay_lease_events(
                self.events,
                worker_ids=sorted(self.worker_lease_limits),
                worker_lease_limits=self.worker_lease_limits,
                worker_concurrency_limits=self.worker_concurrency_limits,
                metric_scale=self.metric_scale,
            )

    def apply_parallel(
        self,
        tasks: Iterable[tuple[str, str, Any]],
        action: Callable[[str, str, Lease, Any], Any],
        *,
        max_workers: int | None = None,
        timeout_ms_each: int | None = None,
    ) -> dict[str, Any]:
        """Run ``action(batch_id, worker_id, lease, extra)`` under leases.

        Results are returned in input order even though grants and execution may
        finish in another order.  Action exceptions cancel no unrelated task;
        all granted leases are released in ``finally`` blocks.
        """
        task_list = list(tasks)
        if not task_list:
            return {}
        limit = max_workers or min(len(task_list), max(1, self.global_limit))

        def run_one(item: tuple[str, str, Any]) -> tuple[str, Any, Lease | None]:
            batch_id, worker_id, extra = item
            lease = self.acquire(batch_id, worker_id, timeout_ms=timeout_ms_each)
            try:
                value = action(batch_id, worker_id, lease, extra)
                return batch_id, value, lease
            finally:
                self.release(lease.lease_id, expected_version=lease.version)

        with ThreadPoolExecutor(max_workers=limit, thread_name_prefix="speaker-diag-lease") as executor:
            futures = [executor.submit(run_one, item) for item in task_list]
            wait_futures(futures)
        results: dict[str, Any] = {}
        for (_batch_id, _worker, _extra), future in zip(task_list, futures):
            _result_batch_id, value, lease = future.result()
            results[_result_batch_id] = {
                "value": value,
                "lease": lease.to_dict(self.origin_ns) if lease else None,
            }
        return results


def replay_lease_events(
    events: Iterable[Mapping[str, Any]],
    *,
    worker_ids: Iterable[str],
    worker_lease_limits: Mapping[str, int],
    worker_concurrency_limits: Mapping[str, int],
    metric_scale: int,
) -> dict[str, Any]:
    active: dict[str, dict[str, Any]] = {}
    per_worker: defaultdict[str, int] = defaultdict(int)
    concurrent: defaultdict[str, int] = defaultdict(int)
    structural_rejections: list[dict[str, Any]] = []
    rejected_operations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: int(item["seq"])):
        event_type = str(event["type"])
        lease_id = event.get("lease_id")
        worker_id = str(event.get("worker_id", ""))
        if event_type == "GRANTED" and lease_id:
            if lease_id in active:
                structural_rejections.append({"seq": event["seq"], "reason": "duplicate_grant", "lease_id": lease_id})
                continue
            active[str(lease_id)] = dict(event)
            per_worker[worker_id] += 1
            concurrent[worker_id] += 1
        elif event_type in ("RELEASED", "EXPIRED") and lease_id:
            if lease_id not in active:
                structural_rejections.append({"seq": event["seq"], "reason": "unknown_lease", "lease_id": lease_id})
                continue
            old = active.pop(str(lease_id))
            per_worker[str(old["worker_id"])] -= 1
            concurrent[str(old["worker_id"])] -= 1
        elif event_type == "RENEWED" and lease_id:
            if lease_id not in active:
                structural_rejections.append({"seq": event["seq"], "reason": "renew_unknown_lease", "lease_id": lease_id})
                continue
            active[str(lease_id)].update(dict(event))
        elif event_type in ("RENEW_REJECTED", "QUEUE_TIMEOUT"):
            rejected_operations.append(dict(event))
        elif event_type in ("QUEUED", "GRANTED", "RELEASED", "EXPIRED", "RENEWED"):
            pass
        else:
            structural_rejections.append({"seq": event.get("seq"), "reason": "unknown_event", "type": event_type})
        steps.append(
            {
                "seq": event.get("seq"),
                "type": event_type,
                "worker_id": worker_id,
                "active_total": len(active),
                "worker_active_leases": {
                    worker: per_worker[worker] for worker in sorted(set(list(per_worker) + list(worker_ids)))
                },
                "worker_watermarks_q": {
                    worker: floor_ratio(per_worker[worker], max(1, worker_lease_limits[worker]), metric_scale)
                    for worker in sorted(worker_ids)
                },
                "worker_concurrency_watermarks_q": {
                    worker: floor_ratio(
                        concurrent[worker], max(1, worker_concurrency_limits[worker]), metric_scale
                    )
                    for worker in sorted(worker_ids)
                },
            }
        )
    if any(value < 0 for value in per_worker.values()) or any(value < 0 for value in concurrent.values()):
        structural_rejections.append({"reason": "negative_counter"})
    return {
        "schema": "speaker-diag-lease-replay-v1",
        "replayable": not structural_rejections,
        "structural_rejections": structural_rejections,
        "rejected_operations": rejected_operations,
        "final_active_leases": sorted(active),
        "final_worker_active_leases": {worker: per_worker[worker] for worker in sorted(worker_ids)},
        "steps": steps,
    }

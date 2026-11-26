from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .errors import CkptError


T = TypeVar("T")


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    task_id: str
    version: int
    granted_ms: int
    expires_ms: int
    status: str = "active"
    granted_ns: int = field(default=0, repr=False)
    expires_ns: int = field(default=0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "task_id": self.task_id,
            "version": self.version,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "status": self.status,
        }


@dataclass
class ScheduledTask:
    seq: int
    shard_id: str
    task_id: str
    fn: Callable[[Lease], T]
    future: Future[T] = field(default_factory=Future)
    lease: Lease | None = None
    wait_start_ms: int = 0
    wait_end_ms: int | None = None
    lease_ready: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    def wait_for_lease(self, timeout: float | None = None) -> Lease:
        if not self.lease_ready.wait(timeout):
            raise TimeoutError("lease grant timed out")
        if self.lease is None:
            raise RuntimeError("task is not leased")
        return self.lease


class LeaseScheduler:
    """Threaded scheduler with global/per-shard lease and concurrency gates."""

    def __init__(
        self,
        *,
        global_lease_limit: int,
        shard_lease_limits: dict[str, int],
        shard_concurrency: dict[str, int],
        ttl_ms: int = 30_000,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if global_lease_limit <= 0:
            raise ValueError("global_lease_limit must be positive")
        self.global_lease_limit = global_lease_limit
        self.shard_lease_limits = dict(shard_lease_limits)
        self.shard_concurrency = dict(shard_concurrency)
        self.ttl_ms = ttl_ms
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._pending: list[ScheduledTask] = []
        self._tasks: list[ScheduledTask] = []
        self._leases: dict[str, Lease] = {}
        self._active_global = 0
        self._active_by_shard = {shard_id: 0 for shard_id in shard_lease_limits}
        self._inflight_by_shard = {shard_id: 0 for shard_id in shard_lease_limits}
        self._wait_records: list[dict[str, Any]] = []
        self._lease_records: list[dict[str, Any]] = []
        self._closed = False
        self._max_active_global = 0
        self._max_active_by_shard = {shard_id: 0 for shard_id in shard_lease_limits}
        self._max_inflight_by_shard = {shard_id: 0 for shard_id in shard_lease_limits}
        total_workers = max(1, sum(self.shard_concurrency.values()))
        self._executor = ThreadPoolExecutor(
            max_workers=total_workers, thread_name_prefix="ckptsim-lease"
        )
        self._janitor = threading.Thread(
            target=self._janitor_loop,
            name="ckptsim-lease-janitor",
            daemon=True,
        )
        self._janitor.start()

    def submit(self, *, seq: int, shard_id: str, task_id: str, fn: Callable[[Lease], T]) -> ScheduledTask:
        if shard_id not in self.shard_lease_limits:
            raise ValueError(f"unknown shard {shard_id!r}")
        task = ScheduledTask(
            seq=seq,
            shard_id=shard_id,
            task_id=task_id,
            fn=fn,
            wait_start_ms=self._clock_ns() // 1_000_000,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("scheduler is closed")
            self._tasks.append(task)
            self._pending.append(task)
            self._pending.sort(key=lambda item: (item.seq, item.shard_id, item.task_id))
            self._dispatch_locked()
        return task

    def _can_dispatch_locked(self, task: ScheduledTask) -> bool:
        return (
            self._active_global < self.global_lease_limit
            and self._active_by_shard[task.shard_id]
            < self.shard_lease_limits[task.shard_id]
            and self._inflight_by_shard[task.shard_id]
            < self.shard_concurrency[task.shard_id]
        )

    def _dispatch_locked(self) -> None:
        made_progress = True
        while made_progress:
            made_progress = False
            for task in list(self._pending):
                if not self._can_dispatch_locked(task):
                    continue
                self._pending.remove(task)
                now_ns = self._clock_ns()
                now_ms = now_ns // 1_000_000
                lease = Lease(
                    lease_id=f"lease-grant-{task.seq}-{task.shard_id}-{task.task_id}",
                    shard_id=task.shard_id,
                    task_id=task.task_id,
                    version=1,
                    granted_ms=now_ms,
                    expires_ms=now_ms + self.ttl_ms,
                    granted_ns=now_ns,
                    expires_ns=now_ns + self.ttl_ms * 1_000_000,
                )
                task.lease = lease
                task.wait_end_ms = now_ms
                self._leases[lease.lease_id] = lease
                self._active_global += 1
                self._active_by_shard[task.shard_id] += 1
                self._inflight_by_shard[task.shard_id] += 1
                self._max_active_global = max(self._max_active_global, self._active_global)
                self._max_active_by_shard[task.shard_id] = max(
                    self._max_active_by_shard[task.shard_id],
                    self._active_by_shard[task.shard_id],
                )
                self._max_inflight_by_shard[task.shard_id] = max(
                    self._max_inflight_by_shard[task.shard_id],
                    self._inflight_by_shard[task.shard_id],
                )
                self._wait_records.append(
                    {
                        "task_id": task.task_id,
                        "shard_id": task.shard_id,
                        "wait_start_ms": task.wait_start_ms,
                        "wait_end_ms": task.wait_end_ms,
                        "wait_ms": task.wait_end_ms - task.wait_start_ms,
                    }
                )
                self._lease_records.append(lease.to_dict())
                task.lease_ready.set()
                self._executor.submit(self._run_task, task, lease)
                made_progress = True

    def _run_task(self, task: ScheduledTask, lease: Lease) -> None:
        try:
            result = task.fn(lease)
        except BaseException as exc:  # propagate to the caller's Future
            if not task.future.done():
                task.future.set_exception(exc)
        else:
            if not task.future.done():
                task.future.set_result(result)
        finally:
            self._finish(task, lease)

    def _finish(self, task: ScheduledTask, lease: Lease) -> None:
        with self._condition:
            if lease.status == "active":
                lease.status = "released"
                self._active_global -= 1
                self._active_by_shard[task.shard_id] -= 1
            self._inflight_by_shard[task.shard_id] -= 1
            task.done.set()
            self._dispatch_locked()
            self._condition.notify_all()

    def _expire_locked(self) -> int:
        now_ns = self._clock_ns()
        expired = 0
        for lease in self._leases.values():
            if lease.status == "active" and now_ns >= lease.expires_ns:
                lease.status = "expired"
                self._active_global -= 1
                self._active_by_shard[lease.shard_id] -= 1
                expired += 1
        if expired:
            self._dispatch_locked()
            self._condition.notify_all()
        return expired

    def _janitor_loop(self) -> None:
        interval = max(0.005, min(0.05, self.ttl_ms / 10_000))
        while True:
            with self._condition:
                if self._closed:
                    return
                self._expire_locked()
                self._condition.wait(timeout=interval)

    def expire_now(self) -> int:
        with self._condition:
            return self._expire_locked()

    def renew(
        self,
        task: ScheduledTask,
        *,
        expected_version: int,
        validate: Callable[[Lease], Any],
    ) -> Any:
        with self._condition:
            lease = task.lease
            if lease is None:
                raise CkptError(
                    "E_LEASE",
                    "lease has not been granted",
                    shard_id=task.shard_id,
                    batch_id=task.task_id,
                )
            now_ns = self._clock_ns()
            if lease.status == "expired" or now_ns >= lease.expires_ns:
                lease.status = "expired"
                raise CkptError(
                    "E_LEASE",
                    "lease has expired",
                    shard_id=task.shard_id,
                    batch_id=task.task_id,
                )
            result = validate(lease)
            if lease.version != expected_version:
                raise CkptError(
                    "E_LEASE",
                    f"lease version mismatch after remote validation: expected {expected_version}, actual {lease.version}",
                    shard_id=task.shard_id,
                    batch_id=task.task_id,
                )
            if lease.status == "expired" or self._clock_ns() >= lease.expires_ns:
                raise CkptError(
                    "E_LEASE",
                    "lease expired while renewal was validated",
                    shard_id=task.shard_id,
                    batch_id=task.task_id,
                )
            lease.version = expected_version + 1
            lease.expires_ns = self._clock_ns() + self.ttl_ms * 1_000_000
            lease.expires_ms = lease.expires_ns // 1_000_000
            return result

    def wait(self, task: ScheduledTask, timeout: float | None = None) -> T:
        return task.future.result(timeout=timeout)

    def wait_all(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            tasks = list(self._tasks)
        for task in tasks:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            task.future.result(timeout=remaining)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "global_lease_limit": self.global_lease_limit,
                "shard_lease_limits": dict(sorted(self.shard_lease_limits.items())),
                "shard_concurrency": dict(sorted(self.shard_concurrency.items())),
                "max_observed_active_leases": self._max_active_global,
                "max_observed_active_leases_by_shard": dict(
                    sorted(self._max_active_by_shard.items())
                ),
                "max_observed_inflight_by_shard": dict(
                    sorted(self._max_inflight_by_shard.items())
                ),
                "wait_records": list(self._wait_records),
                "leases": list(self._lease_records),
            }

    def close(self, wait: bool = True) -> None:
        with self._condition:
            self._closed = True
            if wait:
                deadline = time.monotonic() + 10
                while self._pending or any(
                    value > 0 for value in self._inflight_by_shard.values()
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
            self._condition.notify_all()
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def __enter__(self) -> "LeaseScheduler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close(wait=True)

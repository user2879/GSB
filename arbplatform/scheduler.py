"""Quota-constrained parallel arbitration on a real worker thread pool.

Two real synchronization primitives bound in-flight work:

* a global semaphore sized to the manifest's global concurrency limit;
* one per-environment semaphore sized to that environment's lease capacity.

A task starts only after acquiring both, so at any instant the total in-flight
count and each environment's in-flight count respect their limits. Queueing is
measured with the monotonic clock; no artificial sleep or thread timeout is
used to enforce isolation.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable

from .determinism import now_ms
from .spec import Manifest


@dataclass
class TaskRecord:
    task_id: str
    env_id: str
    view_id: str
    key: str
    enqueued_ms: int
    wait_start_ms: int | None = None
    wait_end_ms: int | None = None
    started_ms: int | None = None
    finished_ms: int | None = None
    result: Any = None
    error: str | None = None

    @property
    def wait_ms(self) -> int | None:
        if self.wait_start_ms is None or self.wait_end_ms is None:
            return None
        return self.wait_end_ms - self.wait_start_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "env_id": self.env_id,
            "view_id": self.view_id,
            "key": self.key,
            "enqueued_ms": self.enqueued_ms,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "wait_ms": self.wait_ms,
            "started_ms": self.started_ms,
            "finished_ms": self.finished_ms,
            "error": self.error,
        }


@dataclass
class SchedulerStats:
    max_global_inflight: int = 0
    max_per_env_inflight: dict[str, int] = field(default_factory=dict)
    records: list[TaskRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_global_inflight": self.max_global_inflight,
            "max_per_env_inflight": dict(sorted(self.max_per_env_inflight.items())),
            "tasks": [r.as_dict() for r in sorted(self.records,
                                                  key=lambda r: r.task_id)],
        }


class QuotaScheduler:
    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        self._global = threading.Semaphore(manifest.global_concurrency)
        self._per_env = {
            env.env_id: threading.Semaphore(env.lease_capacity)
            for env in manifest.environments
        }
        self._stats_lock = threading.Lock()
        self._inflight = 0
        self._env_inflight: dict[str, int] = {e.env_id: 0 for e in manifest.environments}
        self.stats = SchedulerStats()

    def run(self, tasks: list[tuple[str, str, str, Callable[[], Any]]]
            ) -> list[TaskRecord]:
        """Run ``(task_id, env_id, key, callable)`` tasks under both quotas."""
        # The pool may hold more threads than the global quota; the semaphore is
        # the real enforcement mechanism. Pool size is bounded explicitly.
        pool_size = max(1, min(len(tasks), self.manifest.global_concurrency + 2))
        records: dict[str, TaskRecord] = {}
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            for index, (task_id, env_id, view_id, callable_) in enumerate(tasks):
                key = view_id
                record = TaskRecord(
                    task_id=task_id, env_id=env_id, view_id=view_id, key=key,
                    enqueued_ms=now_ms(),
                )
                records[task_id] = record
                futures.append(
                    pool.submit(self._run_one, record, env_id, callable_)
                )
            for future in futures:
                future.result()
        self.stats.records = [records[tid] for tid in sorted(records)]
        return self.stats.records

    def _run_one(self, record: TaskRecord, env_id: str,
                 callable_: Callable[[], Any]) -> None:
        record.wait_start_ms = now_ms()
        with self._global:
            env_sem = self._per_env[env_id]
            env_sem.acquire()
            try:
                record.wait_end_ms = now_ms()
                record.started_ms = now_ms()
                with self._stats_lock:
                    self._inflight += 1
                    self._env_inflight[env_id] += 1
                    self.stats.max_global_inflight = max(
                        self.stats.max_global_inflight, self._inflight)
                    self.stats.max_per_env_inflight[env_id] = max(
                        self.stats.max_per_env_inflight.get(env_id, 0),
                        self._env_inflight[env_id])
                try:
                    record.result = callable_()
                except Exception as exc:  # recorded, never silently swallowed
                    record.error = f"{type(exc).__name__}: {exc}"
                finally:
                    record.finished_ms = now_ms()
                    with self._stats_lock:
                        self._inflight -= 1
                        self._env_inflight[env_id] -= 1
            finally:
                env_sem.release()

"""Bounded parallel aggregation scheduler.

Two real concurrency limits are enforced simultaneously:

* ``global_concurrency`` in-flight window aggregation tasks overall;
* ``per_source_concurrency`` in-flight tasks for any single observation
  source.

The scheduler projects work onto a real :class:`ThreadPoolExecutor` and uses
counting semaphores for every quota decision - there are no artificial
sleeps. Queue wait/start/end timestamps come from the injected monotonic
clock.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .clock import MonotonicClock, SystemClock
from .models import TaskTiming

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AggregationTask(Generic[T]):
    task_id: str
    metric_key: str
    window_index: int
    # Owning observation source for the per-source quota.
    source_id: str
    payload: T


class BoundedScheduler:
    def __init__(self, global_concurrency: int, per_source_concurrency: int,
                 clock: MonotonicClock | None = None):
        if global_concurrency < 1 or per_source_concurrency < 1:
            raise ValueError("concurrency limits must be positive")
        if per_source_concurrency > global_concurrency:
            raise ValueError(
                "per_source_concurrency cannot exceed global_concurrency")
        self._global_limit = global_concurrency
        self._per_source_limit = per_source_concurrency
        self._clock = clock or SystemClock()
        self._max_observed_global = 0
        self._max_observed_source: dict[str, int] = {}

    @property
    def global_limit(self) -> int:
        return self._global_limit

    @property
    def per_source_limit(self) -> int:
        return self._per_source_limit

    def run(
        self,
        tasks: list[AggregationTask[T]],
        worker: Callable[[AggregationTask[T]], object],
    ) -> tuple[list[object], list[TaskTiming]]:
        """Execute all tasks, returning results/timings in task order."""
        if not tasks:
            return [], []

        results: list[object | None] = [None] * len(tasks)
        timings: list[TaskTiming | None] = [None] * len(tasks)
        queue: list[tuple[int, AggregationTask[T]]] = list(enumerate(tasks))
        queue_lock = threading.Lock()
        work_available = threading.Event()
        all_done = threading.Event()
        remaining = len(tasks)

        global_slots = threading.BoundedSemaphore(self._global_limit)
        source_slots = {
            task.source_id: threading.BoundedSemaphore(
                self._per_source_limit)
            for task in tasks
        }
        inflight_global = 0
        inflight_source: dict[str, int] = {}
        max_global = 0
        max_source: dict[str, int] = {}

        pool = ThreadPoolExecutor(max_workers=self._global_limit,
                                  thread_name_prefix="obsflow-agg")
        enqueued_at = [self._clock.now_ms() for _ in tasks]
        wait_started_at = list(enqueued_at)

        def claim_next() -> tuple[int, AggregationTask[T]] | None:
            nonlocal inflight_global, max_global
            with queue_lock:
                if not queue:
                    return None
                for position, (index, task) in enumerate(queue):
                    source_sem = source_slots[task.source_id]
                    if not global_slots.acquire(blocking=False):
                        # Global cap reached; no later task can launch.
                        return None
                    if not source_sem.acquire(blocking=False):
                        global_slots.release()
                        continue
                    del queue[position]
                    inflight_global += 1
                    max_global = max(max_global, inflight_global)
                    count = inflight_source.get(task.source_id, 0) + 1
                    inflight_source[task.source_id] = count
                    max_source[task.source_id] = max(
                        max_source.get(task.source_id, 0), count)
                    now_ms = self._clock.now_ms()
                    timings[index] = TaskTiming(
                        metric_key=task.metric_key,
                        window_index=task.window_index,
                        source_id=task.source_id,
                        enqueued_ms=enqueued_at[index],
                        wait_start_ms=wait_started_at[index],
                        wait_end_ms=now_ms,
                        run_start_ms=now_ms,
                        run_end_ms=0,
                    )
                    return index, task
                return None

        def pool_worker() -> None:
            nonlocal inflight_global, remaining
            while True:
                claimed = claim_next()
                if claimed is None:
                    with queue_lock:
                        if remaining == 0:
                            all_done.set()
                            return
                    # Quota temporarily exhausted; wait for a real release.
                    work_available.wait(timeout=0.05)
                    work_available.clear()
                    continue

                index, task = claimed
                source_sem = source_slots[task.source_id]
                try:
                    results[index] = worker(task)
                    timing = timings[index]
                    assert timing is not None
                    timings[index] = TaskTiming(
                        metric_key=timing.metric_key,
                        window_index=timing.window_index,
                        source_id=timing.source_id,
                        enqueued_ms=timing.enqueued_ms,
                        wait_start_ms=timing.wait_start_ms,
                        wait_end_ms=timing.wait_end_ms,
                        run_start_ms=timing.run_start_ms,
                        run_end_ms=self._clock.now_ms(),
                    )
                finally:
                    source_sem.release()
                    global_slots.release()
                    with queue_lock:
                        inflight_global -= 1
                        inflight_source[task.source_id] = max(
                            0, inflight_source.get(task.source_id, 1) - 1)
                        remaining -= 1
                        if remaining == 0:
                            all_done.set()
                    work_available.set()

        futures = [pool.submit(pool_worker)
                   for _ in range(self._global_limit)]
        try:
            all_done.wait()
        finally:
            pool.shutdown(wait=True, cancel_futures=False)
            for future in futures:
                future.result()

        self._max_observed_global = max_global
        self._max_observed_source = dict(max_source)
        return results, timings  # type: ignore[return-value]

    def observed_max_global(self) -> int:
        return self._max_observed_global

    def observed_max_source(self, source_id: str) -> int:
        return self._max_observed_source.get(source_id, 0)

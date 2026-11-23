"""Bounded-concurrency batch scheduler with real timeout and retry.

Two simultaneous constraints are enforced:
  * at most ``global_limit`` tasks running at any instant;
  * at most ``per_service_limit`` tasks for one service_id at any instant.
When a per-service quota is exhausted the task queues; queue wait start/end are
stamped with a monotonic clock in milliseconds. Each attempt has a real
wall-clock deadline (``task_timeout_ms``); a timed-out attempt is retried up to
``max_attempts`` times. Retried work is pure and deterministic, so retries do
not not change batch semantics.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Callable

log = logging.getLogger("alert_pipeline.scheduler")


class TaskTimeout(Exception):
    pass


@dataclass
class Task:
    task_id: str
    service_id: str
    func: Callable[[], Any]
    payload: dict = None

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}


@dataclass
class BatchMetrics:
    batch_id: int
    started_clock_ms: int
    ended_clock_ms: int = 0
    wait_started_clock_ms: int = 0
    wait_ended_clock_ms: int = 0
    completed_count: int = 0
    submitted_count: int = 0
    peak_concurrency: int = 0
    timed_out_attempts: int = 0
    retried_tasks: int = 0

    def to_record(self):
        return asdict(self)


class MonotonicClock:
    """time.monotonic_ns wrapper returning whole milliseconds."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class Scheduler:
    def __init__(self, global_limit: int, per_service_limit: int,
                 task_timeout_ms: int, max_attempts: int = 3,
                 clock: MonotonicClock | None = None,
                 timeout_task_ids: set[str] | None = None):
        if global_limit < 1 or per_service_limit < 1:
            raise ValueError("concurrency limits must be >= 1")
        self.global_limit = global_limit
        self.per_service_limit = per_service_limit
        self.task_timeout_ms = task_timeout_ms
        self.max_attempts = max_attempts
        self.clock = clock or MonotonicClock()
        # Fault injection: these task ids time out exactly on their 1st attempt.
        self.timeout_task_ids = set(timeout_task_ids or ())
        self._cond = threading.Condition()
        self._global_in_use = 0
        self._service_in_use: dict[str, int] = {}
        self._peak = 0
        self._timed_out_attempts = 0
        self._retried_tasks = 0

    def run_batch(self, batch_id: int, tasks: list[Task]):
        metrics = BatchMetrics(batch_id=batch_id,
                               started_clock_ms=self.clock.now_ms(),
                               submitted_count=len(tasks))
        results: dict[str, tuple[bool, Any]] = {}
        pending = list(tasks)
        wait_started: dict[str, int] = {}
        min_wait_start: int | None = None
        max_wait_end = 0

        def can_start(task: Task) -> bool:
            return (self._global_in_use < self.global_limit
                    and self._service_in_use.get(task.service_id, 0)
                    < self.per_service_limit)

        with ThreadPoolExecutor(max_workers=self.global_limit) as pool:
            with self._cond:
                while pending or self._global_in_use > 0:
                    still_pending: list[Task] = []
                    for task in pending:
                        if can_start(task):
                            if task.task_id in wait_started:
                                max_wait_end = max(
                                    max_wait_end, self.clock.now_ms())
                            self._global_in_use += 1
                            self._service_in_use[task.service_id] = \
                                self._service_in_use.get(task.service_id, 0) + 1
                            assert self._global_in_use <= self.global_limit
                            assert self._service_in_use[task.service_id] \
                                <= self.per_service_limit
                            self._peak = max(self._peak, self._global_in_use)
                            pool.submit(self._run_with_timeout, task, results)
                        else:
                            if task.task_id not in wait_started:
                                wait_started[task.task_id] = \
                                    self.clock.now_ms()
                                if min_wait_start is None:
                                    min_wait_start = \
                                        wait_started[task.task_id]
                            still_pending.append(task)
                    pending = still_pending
                    if pending or self._global_in_use > 0:
                        self._cond.wait(timeout=0.02)

        metrics.ended_clock_ms = self.clock.now_ms()
        metrics.completed_count = sum(1 for ok, _ in results.values() if ok)
        metrics.peak_concurrency = self._peak
        metrics.timed_out_attempts = self._timed_out_attempts
        metrics.retried_tasks = self._retried_tasks
        metrics.wait_started_clock_ms = min_wait_start or 0
        metrics.wait_ended_clock_ms = max_wait_end

        failures = [tid for tid, (ok, _) in results.items() if not ok]
        if failures:
            raise TaskTimeout(
                f"batch {batch_id}: task(s) {sorted(failures)} exhausted "
                f"{self.max_attempts} attempts")
        return [results[task.task_id][1] for task in tasks], metrics

    def _run_with_timeout(self, task: Task,
                          results: dict[str, tuple[bool, Any]]) -> None:
        try:
            timed_out = True
            for attempt in range(1, self.max_attempts + 1):
                timed_out, value = self._attempt(task, attempt)
                if not timed_out:
                    results[task.task_id] = (True, value)
                    return
                self._timed_out_attempts += 1
                if attempt < self.max_attempts:
                    self._retried_tasks += 1
                    log.warning("task %s timed out (attempt %s), retrying",
                                task.task_id, attempt)
            results[task.task_id] = (False, TaskTimeout(task.task_id))
        finally:
            with self._cond:
                self._global_in_use -= 1
                self._service_in_use[task.service_id] -= 1
                self._cond.notify_all()

    def _attempt(self, task: Task, attempt: int) -> tuple[bool, Any]:
        """Run one attempt with a real deadline. Returns (timed_out, value)."""
        box: dict = {}
        done = threading.Event()

        def worker() -> None:
            try:
                box["value"] = task.func()
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                box["error"] = exc
            finally:
                done.set()

        injected = (attempt == 1
                    and task.task_id in self.timeout_task_ids)
        if injected:
            # Fault injection: the worker itself outlives the real deadline.
            original = task.func

            def slow_func():
                time.sleep(self.task_timeout_ms / 1000.0 + 0.1)
                return original()

            worker_ref = slow_func
            box_local = box
            done_local = done

            def slow_worker():
                try:
                    box_local["value"] = worker_ref()
                except Exception as exc:  # noqa: BLE001
                    box_local["error"] = exc
                finally:
                    done_local.set()

            thread = threading.Thread(target=slow_worker,
                                      name=f"task-{task.task_id}", daemon=True)
        else:
            thread = threading.Thread(
                target=worker, name=f"task-{task.task_id}", daemon=True)
        thread.start()
        if not done.wait(timeout=self.task_timeout_ms / 1000.0):
            return True, None
        if "error" in box:
            raise box["error"]
        return False, box["value"]

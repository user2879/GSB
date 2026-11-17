"""Block-level parallel scan scheduler under three constraints:

1. global worker parallelism (``max_parallelism``);
2. global decoded-memory budget (``memory_budget_bytes``);
3. per-segment in-flight concurrency quota (``per_segment_concurrency``).

The scheduler only owns resource accounting and lifecycle timestamps. Block
work is supplied as a callback so scheduling policy is independent of segment
decoding.  Result ordering is handled by the caller from deterministic block
coordinates; thread completion timing therefore never influences results.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

from .errors import BudgetError
from .model import BlockId, BudgetParams, Encoding, TaskRecord


@dataclass
class TaskSpec:
    block: BlockId
    encoding: Encoding
    memory_bytes: int
    rows: int


WorkerCallback = Callable[[TaskRecord], int]
"""Callback returns the number of rows matched inside this block."""


class Scheduler:
    def __init__(
        self,
        budget: BudgetParams,
        budget_version: int,
        worker: WorkerCallback,
        clock: Callable[[], float] = lambda: time.monotonic() * 1000.0,
    ):
        self._validate(budget)
        self.budget = budget
        self.budget_version = budget_version
        self._worker = worker
        self._clock = clock

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: Deque[TaskRecord] = deque()
        self._running = 0
        self._memory_used = 0
        self._segment_running: Dict[str, int] = {}
        self._peak_memory = 0
        self._memory_time_integral = 0.0
        self._cancelled = False
        self._threads: List[threading.Thread] = []
        self.task_errors: List[BaseException] = []

    @staticmethod
    def _validate(budget: BudgetParams) -> None:
        if budget.memory_budget_bytes <= 0:
            raise BudgetError(
                "decoded-memory budget must be positive",
                {"memory_budget_bytes": budget.memory_budget_bytes},
            )
        if budget.max_parallelism <= 0:
            raise BudgetError(
                "parallelism must be positive",
                {"max_parallelism": budget.max_parallelism},
            )
        if budget.per_segment_concurrency <= 0:
            raise BudgetError(
                "per-segment concurrency must be positive",
                {"per_segment_concurrency": budget.per_segment_concurrency},
            )

    def run(self, specs: List[TaskSpec]) -> List[TaskRecord]:
        now = self._clock()
        records: List[TaskRecord] = []
        for index, spec in enumerate(specs):
            if spec.memory_bytes > self.budget.memory_budget_bytes:
                raise BudgetError(
                    "single block decode memory exceeds global budget; "
                    "no legal schedule exists",
                    {
                        "block": spec.block.to_dict(),
                        "task_memory_bytes": spec.memory_bytes,
                        "budget_bytes": self.budget.memory_budget_bytes,
                    },
                )
            record = TaskRecord(
                task_id=index,
                block=spec.block,
                encoding=spec.encoding,
                memory_bytes=spec.memory_bytes,
                rows=spec.rows,
                enqueue_time_ms=now,
                wait_start_ms=now,
                budget_version=self.budget_version,
            )
            records.append(record)
            self._pending.append(record)

        with self._cond:
            while self._pending or self._running:
                self._dispatch_locked()
                if not self._running:
                    if self._cancelled:
                        break
                    # Nothing runnable despite pending work: every pending task
                    # fits the budget (pre-checked), so this cannot deadlock.
                    self._cond.wait(timeout=0.1)
                    continue
                self._cond.wait()

        for thread in self._threads:
            thread.join()
        if self.task_errors:
            raise self.task_errors[0]
        return records

    def cancel(self) -> int:
        """Mark all not-started tasks abandoned (stale generation guard)."""
        with self._cond:
            self._cancelled = True
            abandoned = 0
            remaining = list(self._pending)
            self._pending.clear()
            for record in remaining:
                record.status = "abandoned"
                record.end_ms = self._clock()
                abandoned += 1
            self._cond.notify_all()
        return abandoned

    @property
    def peak_memory_bytes(self) -> int:
        return self._peak_memory

    @property
    def memory_time_integral_ms(self) -> float:
        return self._memory_time_integral

    def _dispatch_locked(self) -> None:
        if self._cancelled:
            return
        still_pending: Deque[TaskRecord] = deque()
        while self._pending:
            record = self._pending.popleft()
            segment = record.block.segment
            seg_active = self._segment_running.get(segment, 0)
            fits = (
                self._running < self.budget.max_parallelism
                and self._memory_used + record.memory_bytes
                <= self.budget.memory_budget_bytes
                and seg_active < self.budget.per_segment_concurrency
            )
            if not fits:
                still_pending.append(record)
                continue
            self._reserve_locked(record)
            thread = threading.Thread(
                target=self._run_task, args=(record,), name=f"scan-worker-{record.task_id}"
            )
            record.thread_name = thread.name
            self._threads.append(thread)
            thread.start()
        # Preserve original FIFO order for the tasks that could not start.
        self._pending = still_pending

    def _reserve_locked(self, record: TaskRecord) -> None:
        record.status = "running"
        record.start_ms = self._clock()
        self._running += 1
        self._memory_used += record.memory_bytes
        self._segment_running[record.block.segment] = (
            self._segment_running.get(record.block.segment, 0) + 1
        )
        self._peak_memory = max(self._peak_memory, self._memory_used)

    def _run_task(self, record: TaskRecord) -> None:
        error: Optional[BaseException] = None
        matched = 0
        try:
            matched = self._worker(record)
        except BaseException as exc:  # surface to dispatcher; never hang pool
            error = exc
        with self._cond:
            record.end_ms = self._clock()
            record.matched_rows = matched
            self._memory_time_integral += record.memory_bytes * max(
                0.0, record.end_ms - record.start_ms
            )
            if error is None:
                record.status = "done"
            else:
                record.status = "failed"
                record.__dict__["error"] = error
                self.task_errors.append(error)
            self._running -= 1
            self._memory_used -= record.memory_bytes
            active = self._segment_running.get(record.block.segment, 0) - 1
            if active <= 0:
                self._segment_running.pop(record.block.segment, None)
            else:
                self._segment_running[record.block.segment] = active
            self._cond.notify_all()
        if error is not None:
            raise error

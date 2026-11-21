from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

from .canonical import monotonic_ms, rounded_ms
from .errors import TaskTimedOut
from .model import Budget
from .budget import Allocation
from .visibility import VisibilityDecision


@dataclass(frozen=True)
class TaskTrace:
    query_id: str
    read_point_id: str
    row_key: str
    queue_enter_mono_ms: float
    queue_exit_mono_ms: float
    started_mono_ms: float
    ended_mono_ms: float
    wait_ms: float
    worker_delay_ms: float
    duration_ms: float
    timeout_ms: int
    status: str
    decision: VisibilityDecision | None
    error: str | None = None

    def as_dict(self, *, include_decision: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "queue_enter_mono_ms": self.queue_enter_mono_ms,
            "queue_exit_mono_ms": self.queue_exit_mono_ms,
            "started_mono_ms": self.started_mono_ms,
            "ended_mono_ms": self.ended_mono_ms,
            "wait_ms": self.wait_ms,
            "worker_delay_ms": self.worker_delay_ms,
            "duration_ms": self.duration_ms,
            "timeout_ms": self.timeout_ms,
            "status": self.status,
            "decision_outcome": self.decision.outcome if self.decision else None,
            "error": self.error,
        }
        if include_decision and self.decision:
            result["decision"] = self.decision.as_dict()
        return result


class QuotaScheduler:
    def __init__(self, budget: Budget):
        self.budget = budget
        self._lock = threading.Lock()
        self.max_observed_global = 0
        self.max_observed_per_read_point: dict[str, int] = {}

    def run(
        self,
        allocations: tuple[Allocation, ...],
        operation: Callable[[Allocation, int], VisibilityDecision],
        *,
        on_submit: Callable[[int, Allocation], None] | None = None,
    ) -> tuple[TaskTrace, ...]:
        queue_enter = monotonic_ms()
        pending = list(allocations)
        active_by_rp: Counter[str] = Counter()
        active_total = 0
        futures: dict[Future[TaskTrace], Allocation] = {}
        traces: dict[str, TaskTrace] = {}
        submitted_count = 0

        def execute(allocation: Allocation, admitted_mono_ms: float) -> TaskTrace:
            started_ns = __import__("time").monotonic_ns()
            started = started_ns / 1_000_000.0
            deadline_ns = started_ns + self.budget.task_timeout_ms * 1_000_000
            decision: VisibilityDecision | None = None
            error: str | None = None
            status = "OK"
            try:
                decision = operation(allocation, deadline_ns)
                ended_ns = __import__("time").monotonic_ns()
                if ended_ns >= deadline_ns:
                    status = "TIMEOUT"
                    decision = None
                    error = "real elapsed time reached the monotonic deadline"
            except TaskTimedOut as exc:
                ended_ns = __import__("time").monotonic_ns()
                status = "TIMEOUT"
                error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                ended_ns = __import__("time").monotonic_ns()
                status = "ERROR"
                error = f"{type(exc).__name__}: {exc}"
            ended = ended_ns / 1_000_000.0
            with self._lock:
                self.max_observed_global = max(self.max_observed_global, active_total)
                self.max_observed_per_read_point[allocation.read_point_id] = max(
                    self.max_observed_per_read_point.get(allocation.read_point_id, 0),
                    active_by_rp[allocation.read_point_id],
                )
            return TaskTrace(
                query_id=allocation.query_id,
                read_point_id=allocation.read_point_id,
                row_key=allocation.row_key,
                queue_enter_mono_ms=rounded_ms(queue_enter),
                queue_exit_mono_ms=rounded_ms(admitted_mono_ms),
                started_mono_ms=rounded_ms(started),
                ended_mono_ms=rounded_ms(ended),
                wait_ms=rounded_ms(admitted_mono_ms - queue_enter),
                worker_delay_ms=rounded_ms(started - admitted_mono_ms),
                duration_ms=rounded_ms(ended - started),
                timeout_ms=self.budget.task_timeout_ms,
                status=status,
                decision=decision,
                error=error,
            )

        with ThreadPoolExecutor(
            max_workers=self.budget.global_concurrency,
            thread_name_prefix="mvcc-visibility",
        ) as executor:
            while pending or futures:
                admitted = True
                while admitted:
                    admitted = False
                    for index, allocation in enumerate(pending):
                        if active_total >= self.budget.global_concurrency:
                            break
                        if active_by_rp[allocation.read_point_id] >= self.budget.per_read_point_concurrency:
                            continue
                        pending.pop(index)
                        active_by_rp[allocation.read_point_id] += 1
                        active_total += 1
                        with self._lock:
                            self.max_observed_global = max(self.max_observed_global, active_total)
                            self.max_observed_per_read_point[allocation.read_point_id] = max(
                                self.max_observed_per_read_point.get(allocation.read_point_id, 0),
                                active_by_rp[allocation.read_point_id],
                            )
                        admitted_mono_ms = monotonic_ms()
                        future = executor.submit(execute, allocation, admitted_mono_ms)
                        futures[future] = allocation
                        submitted_count += 1
                        if on_submit is not None:
                            on_submit(submitted_count, allocation)
                        admitted = True
                        break

                if not futures:
                    if pending:
                        raise RuntimeError("scheduler cannot admit pending work under configured quotas")
                    break
                done, _ = wait(tuple(futures), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    allocation = futures.pop(future)
                    active_total -= 1
                    active_by_rp[allocation.read_point_id] -= 1
                    traces[allocation.query_id] = future.result()

        ordered = tuple(traces[allocation.query_id] for allocation in allocations)
        return ordered

"""Dual-quota parallel dispatch backed by a real thread pool."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import SchedulerError


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    read_point_id: str
    fn: Callable[[], Any] = field(repr=False, compare=False)


@dataclass
class _TaskState:
    task: ScheduledTask
    queued_at_ms: float
    wait_start_ms: float | None = None
    wait_end_ms: float | None = None
    started_at_ms: float | None = None
    finished_at_ms: float | None = None
    thread_name: str | None = None
    thread_id: int | None = None
    blocked_by_global: bool = False
    blocked_by_read_point: bool = False
    result: Any = None
    error: BaseException | None = None

    def trace_dict(self) -> dict[str, Any]:
        wait_ms = 0.0
        if self.wait_start_ms is not None and self.wait_end_ms is not None:
            wait_ms = max(0.0, self.wait_end_ms - self.wait_start_ms)
        duration_ms = 0.0
        if self.started_at_ms is not None and self.finished_at_ms is not None:
            duration_ms = max(0.0, self.finished_at_ms - self.started_at_ms)
        return {
            "task_id": self.task.task_id,
            "read_point_id": self.task.read_point_id,
            "thread_name": self.thread_name,
            "thread_id": self.thread_id,
            "queued_at_ms": round(self.queued_at_ms, 3),
            "quota_wait_start_ms": (
                round(self.wait_start_ms, 3) if self.wait_start_ms is not None else None
            ),
            "quota_wait_end_ms": (
                round(self.wait_end_ms, 3) if self.wait_end_ms is not None else None
            ),
            "quota_wait_ms": round(wait_ms, 3),
            "started_at_ms": (
                round(self.started_at_ms, 3) if self.started_at_ms is not None else None
            ),
            "finished_at_ms": (
                round(self.finished_at_ms, 3) if self.finished_at_ms is not None else None
            ),
            "duration_ms": round(duration_ms, 3),
            "blocked_by_global": self.blocked_by_global,
            "blocked_by_read_point": self.blocked_by_read_point,
            "errored": self.error is not None,
        }


@dataclass(frozen=True)
class ScheduleReport:
    traces: tuple[dict[str, Any], ...]
    results: dict[str, Any]
    budget: dict[str, Any]
    observed_peaks: dict[str, Any]
    totals: dict[str, Any]
    monotonic_clock: str = "time.monotonic_ns/1_000_000"

    def to_dict(self) -> dict[str, Any]:
        return {
            "monotonic_clock": self.monotonic_clock,
            "budget": self.budget,
            "observed_peaks": self.observed_peaks,
            "totals": self.totals,
            "traces": list(self.traces),
        }


class _ActualConcurrency:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._peak = 0
        self._by_read_point: dict[str, int] = defaultdict(int)
        self._peak_by_read_point: dict[str, int] = defaultdict(int)

    def enter(self, read_point_id: str) -> None:
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
            self._by_read_point[read_point_id] += 1
            self._peak_by_read_point[read_point_id] = max(
                self._peak_by_read_point[read_point_id],
                self._by_read_point[read_point_id],
            )

    def leave(self, read_point_id: str) -> None:
        with self._lock:
            self._active -= 1
            self._by_read_point[read_point_id] -= 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "global": self._peak,
                "per_read_point": {
                    key: self._peak_by_read_point[key]
                    for key in sorted(self._peak_by_read_point)
                },
            }


def run_scheduled_tasks(
    tasks: Iterable[ScheduledTask],
    *,
    global_concurrency: int,
    per_read_point_concurrency: int,
) -> ScheduleReport:
    if global_concurrency <= 0 or per_read_point_concurrency <= 0:
        raise SchedulerError("并发预算必须为正整数")
    ordered = sorted(tasks, key=lambda item: item.task_id)
    task_ids = [item.task_id for item in ordered]
    if len(task_ids) != len(set(task_ids)):
        raise SchedulerError("任务标识必须唯一")

    origin_ns = time.monotonic_ns()

    def elapsed_ms() -> float:
        return (time.monotonic_ns() - origin_ns) / 1_000_000.0

    active_global = 0
    active_by_read_point: dict[str, int] = defaultdict(int)
    states: dict[str, _TaskState] = {
        task.task_id: _TaskState(task=task, queued_at_ms=elapsed_ms()) for task in ordered
    }
    pending: deque[_TaskState] = deque(states[task_id] for task_id in task_ids)
    futures: dict[Future[Any], _TaskState] = {}
    actual = _ActualConcurrency()

    def worker(state: _TaskState) -> _TaskState:
        actual.enter(state.task.read_point_id)
        state.thread_name = threading.current_thread().name
        state.thread_id = threading.get_ident()
        state.started_at_ms = elapsed_ms()
        try:
            state.result = state.task.fn()
        except BaseException as exc:  # captured and reported after all running tasks finish
            state.error = exc
        finally:
            state.finished_at_ms = elapsed_ms()
            actual.leave(state.task.read_point_id)
        return state

    with ThreadPoolExecutor(
        max_workers=global_concurrency, thread_name_prefix="mvcc-worker"
    ) as executor:
        while pending or futures:
            startable = True
            while startable:
                startable = False
                remaining: deque[_TaskState] = deque()
                while pending:
                    state = pending.popleft()
                    read_point_id = state.task.read_point_id
                    if active_global >= global_concurrency:
                        state.blocked_by_global = True
                        if state.wait_start_ms is None:
                            state.wait_start_ms = elapsed_ms()
                        remaining.append(state)
                        continue
                    if active_by_read_point[read_point_id] >= per_read_point_concurrency:
                        state.blocked_by_read_point = True
                        if state.wait_start_ms is None:
                            state.wait_start_ms = elapsed_ms()
                        remaining.append(state)
                        continue
                    if state.wait_start_ms is not None:
                        state.wait_end_ms = elapsed_ms()
                    active_global += 1
                    active_by_read_point[read_point_id] += 1
                    future = executor.submit(worker, state)
                    futures[future] = state
                    startable = True
                pending = remaining

            if not futures:
                if pending:
                    raise SchedulerError(
                        "调度器无法推进：存在任务但无运行任务或可用配额",
                        details={"pending": [state.task.task_id for state in pending]},
                    )
                break

            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                state = futures.pop(future)
                future.result()
                active_global -= 1
                active_by_read_point[state.task.read_point_id] -= 1

    errors = [
        states[task_id]
        for task_id in task_ids
        if states[task_id].error is not None
    ]
    if errors:
        first = errors[0]
        raise SchedulerError(
            f"任务执行失败: {first.task.task_id}",
            details={
                "task_id": first.task.task_id,
                "read_point_id": first.task.read_point_id,
                "error_type": type(first.error).__name__ if first.error else None,
                "error": str(first.error) if first.error else None,
                "failed_task_ids": [state.task.task_id for state in errors],
            },
        ) from first.error

    traces = tuple(states[task_id].trace_dict() for task_id in task_ids)
    results = {task_id: states[task_id].result for task_id in task_ids}
    peak = actual.snapshot()
    max_wait = max((trace["quota_wait_ms"] for trace in traces), default=0.0)
    blocked = sum(1 for trace in traces if trace["quota_wait_ms"] > 0.0)
    tasks_by_read_point = Counter(trace["read_point_id"] for trace in traces)
    blocked_by_read_point = Counter(
        trace["read_point_id"] for trace in traces if trace["quota_wait_ms"] > 0.0
    )
    totals = {
        "task_count": len(task_ids),
        "blocked_task_count": blocked,
        "max_quota_wait_ms": round(max_wait, 3),
        "thread_count": len({trace["thread_id"] for trace in traces if trace["thread_id"]}),
    }
    budget = {
        "global_concurrency": global_concurrency,
        "per_read_point_concurrency": per_read_point_concurrency,
        "allocation": {
            "tasks_by_read_point": {
                key: tasks_by_read_point[key] for key in sorted(tasks_by_read_point)
            },
            "blocked_tasks_by_read_point": {
                key: blocked_by_read_point[key] for key in sorted(blocked_by_read_point)
            },
        },
        "constraints_ok": (
            peak["global"] <= global_concurrency
            and all(
                value <= per_read_point_concurrency
                for value in peak["per_read_point"].values()
            )
        ),
    }
    return ScheduleReport(
        traces=traces,
        results=results,
        budget=budget,
        observed_peaks=peak,
        totals=totals,
    )

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import BudgetError, VScanError
from .executor import (
    BatchComputation,
    BatchResult,
    execute_batch_for_query,
)
from .planner import BatchPlan
from .spec import QuerySpec, ScanSpec
from .store import StoreState


class SchedulerError(VScanError):
    code = "E_SCHEDULE"


@dataclass(frozen=True)
class SchedulerTask:
    plan: BatchPlan
    query: QuerySpec
    fingerprint: str
    ready: bool = True
    dependencies: tuple[tuple[str, str, int], ...] = ()

    @property
    def key(self) -> tuple[str, str, int]:
        return self.plan.key


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class SchedulerStats:
    max_observed_inflight: int
    max_observed_decoded_bytes: int
    max_observed_segment_decoded_bytes: dict[str, int]
    dispatched_count: int
    queue_overlap_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_observed_inflight": self.max_observed_inflight,
            "max_observed_decoded_bytes": self.max_observed_decoded_bytes,
            "max_observed_segment_decoded_bytes": self.max_observed_segment_decoded_bytes,
            "dispatched_count": self.dispatched_count,
            "queue_overlap_count": self.queue_overlap_count,
        }


def run_batches(
    spec: ScanSpec,
    store: StoreState,
    data_root: str | Path,
    tasks: Iterable[SchedulerTask],
    *,
    on_commit: Callable[[BatchResult], None] | None = None,
    executor_fn: Callable[[SchedulerTask], BatchComputation] | None = None,
) -> tuple[tuple[BatchResult, ...], SchedulerStats]:
    pending = sorted(
        tuple(tasks),
        key=lambda task: (task.plan.segment_id, task.plan.batch_no, task.plan.query_id),
    )
    if not pending:
        return (), SchedulerStats(0, 0, {}, 0, 0)
    if len({task.key for task in pending}) != len(pending):
        raise SchedulerError("duplicate batch task in one scheduling round")

    memory_budget = spec.execution.segment_decode_memory_bytes
    concurrency = spec.execution.max_concurrency
    raw_run = executor_fn or (
        lambda task: execute_batch_for_query(
            spec, store, data_root, task.query, task.plan
        )
    )
    def run(task: SchedulerTask) -> tuple[BatchComputation, str]:
        computation = raw_run(task)
        return computation, threading.current_thread().name

    queue_enter = {task.key: _monotonic_ms() for task in pending}
    results: list[BatchResult] = []
    future_map: dict[Future[tuple[BatchComputation, str]], SchedulerTask] = {}
    dispatch_times: dict[tuple[str, str, int], int] = {}
    segment_usage: dict[str, int] = {}
    total_usage = 0
    max_inflight = 0
    max_total = 0
    max_segment: dict[str, int] = {}
    dispatched: set[tuple[str, str, int]] = set()
    completed: set[tuple[str, str, int]] = set()
    queue_overlap = 0

    def can_admit(task: SchedulerTask) -> bool:
        amount = task.plan.admission_decoded_bytes
        if total_usage + amount > memory_budget:
            return False
        if segment_usage.get(task.plan.segment_id, 0) + amount > memory_budget:
            return False
        return True

    def submit(task: SchedulerTask) -> None:
        nonlocal total_usage
        if task.key in dispatched:
            raise SchedulerError("a batch may execute at most once per scheduling round")
        amount = task.plan.admission_decoded_bytes
        if amount > memory_budget:
            raise BudgetError(
                "single batch admission exceeds segment decode memory budget",
                query_id=task.plan.query_id,
                segment_id=task.plan.segment_id,
                batch_no=task.plan.batch_no,
                required_bytes=amount,
                budget_bytes=memory_budget,
            )
        dispatched.add(task.key)
        total_usage += amount
        segment_usage[task.plan.segment_id] = segment_usage.get(task.plan.segment_id, 0) + amount
        max_segment[task.plan.segment_id] = max(
            max_segment.get(task.plan.segment_id, 0), segment_usage[task.plan.segment_id]
        )
        future = pool.submit(run, task)
        future_map[future] = task
        dispatch_times[task.key] = _monotonic_ms()

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="vscan-batch") as pool:
        while pending or future_map:
            remaining: list[SchedulerTask] = []
            for task in pending:
                if len(future_map) >= concurrency:
                    remaining.append(task)
                    continue
                dependencies_ready = all(
                    dependency in completed for dependency in task.dependencies
                )
                if not task.ready or not dependencies_ready:
                    remaining.append(task)
                    continue
                if not can_admit(task):
                    remaining.append(task)
                    continue
                submit(task)
            pending = remaining
            max_inflight = max(max_inflight, len(future_map))
            max_total = max(max_total, total_usage)
            if not future_map:
                if not pending:
                    break
                first = pending[0]
                amount = first.plan.admission_decoded_bytes
                if amount > memory_budget:
                    raise BudgetError(
                        "batch admission exceeds decode memory budget",
                        query_id=first.plan.query_id,
                        segment_id=first.plan.segment_id,
                        batch_no=first.plan.batch_no,
                        required_bytes=amount,
                        budget_bytes=memory_budget,
                    )
                unresolved = sorted(
                    dependency
                    for dependency in first.dependencies
                    if dependency not in completed
                )
                raise SchedulerError(
                    "batch cannot make progress because dependencies, readiness or memory admission blocked the queue",
                    query_id=first.plan.query_id,
                    segment_id=first.plan.segment_id,
                    batch_no=first.plan.batch_no,
                    unresolved_dependencies=unresolved,
                    ready=first.ready,
                    required_bytes=amount,
                    budget_bytes=memory_budget,
                )
            done, _ = wait(tuple(future_map), timeout=None, return_when=FIRST_COMPLETED)
            if len(future_map) > 1:
                queue_overlap += 1
            for future in done:
                task = future_map.pop(future)
                amount = task.plan.admission_decoded_bytes
                total_usage -= amount
                segment_usage[task.plan.segment_id] -= amount
                computation, worker_thread = future.result()
                dispatch_ms = dispatch_times.pop(task.key)
                wait_start = queue_enter[task.key]
                result = BatchResult(
                    plan=task.plan,
                    computation=computation,
                    fingerprint=task.fingerprint,
                    wait_start_ms=wait_start,
                    wait_end_ms=dispatch_ms,
                    wait_ms=max(0, dispatch_ms - wait_start),
                    worker_thread=worker_thread,
                    dispatch_count=1,
                )
                if on_commit is not None:
                    on_commit(result)
                results.append(result)
                completed.add(result.key)

    ordered = tuple(sorted(results, key=lambda result: result.key))
    return ordered, SchedulerStats(
        max_observed_inflight=max_inflight,
        max_observed_decoded_bytes=max_total,
        max_observed_segment_decoded_bytes=dict(sorted(max_segment.items())),
        dispatched_count=len(dispatched),
        queue_overlap_count=queue_overlap,
    )

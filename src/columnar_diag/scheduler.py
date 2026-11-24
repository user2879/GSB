from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .errors import AppError
from .models import BudgetConfig, ScanTask


def _now_ms() -> float:
    return time.monotonic_ns() / 1_000_000.0


@dataclass(frozen=True)
class ScheduledOutcome:
    task: ScanTask
    queue_started_ms: float
    wait_ended_ms: float
    run_started_ms: float
    run_ended_ms: float
    duration_ms: float
    value: Any


@dataclass(frozen=True)
class ScheduleReport:
    outcomes: tuple[ScheduledOutcome, ...]
    metrics: dict[str, Any]


def validate_budget(budget: BudgetConfig, tasks: Iterable[ScanTask] = ()) -> None:
    if budget.global_parallelism <= 0:
        raise AppError("E_BUDGET", "global_parallelism 必须为正整数", field="global_parallelism")
    if budget.column_memory_budget_bytes <= 0:
        raise AppError(
            "E_BUDGET",
            "column_memory_budget_bytes 必须为正整数",
            field="column_memory_budget_bytes",
        )
    if budget.segment_concurrency <= 0:
        raise AppError("E_BUDGET", "segment_concurrency 必须为正整数", field="segment_concurrency")
    for task in tasks:
        if task.decode_memory_bytes > budget.column_memory_budget_bytes:
            raise AppError(
                "E_BUDGET",
                "单个块解码内存超过每列预算，任务无法被调度",
                block_id=task.block_id,
                field="decode_memory_bytes",
                details={
                    "task_id": task.task_id,
                    "required": task.decode_memory_bytes,
                    "budget": budget.column_memory_budget_bytes,
                },
            )


def run_scheduled(
    tasks: Iterable[ScanTask],
    worker: Callable[[ScanTask], Any],
    budget: BudgetConfig,
    *,
    worker_count: int | None = None,
) -> ScheduleReport:
    """Schedule block tasks under all three resource gates.

    Admission happens synchronously on the caller thread. Worker threads only
    execute an already-admitted task, so no worker can bypass a budget gate.
    """

    ordered = tuple(sorted(tasks, key=lambda task: task.block_ordinal))
    validate_budget(budget, ordered)
    if not ordered:
        return ScheduleReport(
            outcomes=(),
            metrics={
                "tasks": 0,
                "peak_global_tasks": 0,
                "peak_segment_tasks": {},
                "peak_column_tasks": {},
                "peak_column_memory_bytes": {},
                "peak_total_memory_bytes": 0,
                "max_queue_wait_ms": 0.0,
                "worker_threads": 0,
            },
        )
    pool_size = worker_count or budget.global_parallelism
    if pool_size < budget.global_parallelism:
        raise AppError(
            "E_BUDGET",
            "线程池容量不得小于全局并行度预算",
            field="global_parallelism",
        )
    queue_started = {task.task_id: _now_ms() for task in ordered}
    pending = list(ordered)
    running: dict[Future[Any], tuple[ScanTask, float, float]] = {}
    outcomes: list[ScheduledOutcome] = []
    segment_running: dict[str, int] = defaultdict(int)
    column_running: dict[str, int] = defaultdict(int)
    column_memory: dict[str, int] = defaultdict(int)
    total_memory = 0
    peak_global = 0
    peak_segment: dict[str, int] = defaultdict(int)
    peak_column_tasks: dict[str, int] = defaultdict(int)
    peak_column_memory: dict[str, int] = defaultdict(int)
    peak_total_memory = 0
    max_queue_wait = 0.0
    executor = ThreadPoolExecutor(
        max_workers=pool_size,
        thread_name_prefix="columnar-scan",
    )
    try:
        while pending or running:
            admitted = False
            for task in list(pending):
                if len(running) >= budget.global_parallelism:
                    break
                if segment_running[task.segment_id] >= budget.segment_concurrency:
                    continue
                if (
                    column_memory[task.column] + task.decode_memory_bytes
                    > budget.column_memory_budget_bytes
                ):
                    continue
                wait_ended = _now_ms()
                max_queue_wait = max(max_queue_wait, wait_ended - queue_started[task.task_id])
                pending.remove(task)
                run_started = _now_ms()
                future = executor.submit(worker, task)
                running[future] = (task, run_started, wait_ended)
                segment_running[task.segment_id] += 1
                column_running[task.column] += 1
                column_memory[task.column] += task.decode_memory_bytes
                total_memory += task.decode_memory_bytes
                peak_global = max(peak_global, len(running))
                peak_segment[task.segment_id] = max(
                    peak_segment[task.segment_id], segment_running[task.segment_id]
                )
                peak_column_tasks[task.column] = max(
                    peak_column_tasks[task.column], column_running[task.column]
                )
                peak_column_memory[task.column] = max(
                    peak_column_memory[task.column], column_memory[task.column]
                )
                peak_total_memory = max(peak_total_memory, total_memory)
                admitted = True
            if not running:
                if pending:
                    blocked = pending[0]
                    raise AppError(
                        "E_BUDGET",
                        "没有任务满足调度约束，无法继续",
                        block_id=blocked.block_id,
                        field="scheduler",
                        details={"task_id": blocked.task_id},
                    )
                break
            if admitted:
                continue
            done, _ = wait(running.keys(), timeout=0.05, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                task, run_started, wait_ended = running.pop(future)
                run_ended = _now_ms()
                segment_running[task.segment_id] -= 1
                column_running[task.column] -= 1
                column_memory[task.column] -= task.decode_memory_bytes
                total_memory -= task.decode_memory_bytes
                try:
                    value = future.result()
                except Exception as error:
                    raise AppError(
                        "E_SCAN",
                        f"扫描任务失败: {error}",
                        block_id=task.block_id,
                        field="worker",
                        details={"task_id": task.task_id},
                    ) from error
                outcomes.append(
                    ScheduledOutcome(
                        task=task,
                        queue_started_ms=queue_started[task.task_id],
                        wait_ended_ms=wait_ended,
                        run_started_ms=run_started,
                        run_ended_ms=run_ended,
                        duration_ms=run_ended - run_started,
                        value=value,
                    )
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    outcomes.sort(key=lambda outcome: outcome.task.block_ordinal)
    return ScheduleReport(
        outcomes=tuple(outcomes),
        metrics={
            "tasks": len(outcomes),
            "peak_global_tasks": peak_global,
            "peak_segment_tasks": dict(sorted(peak_segment.items())),
            "peak_column_tasks": dict(sorted(peak_column_tasks.items())),
            "peak_column_memory_bytes": dict(sorted(peak_column_memory.items())),
            "peak_total_memory_bytes": peak_total_memory,
            "max_queue_wait_ms": max_queue_wait,
            "worker_threads": pool_size,
            "global_parallelism": budget.global_parallelism,
            "column_memory_budget_bytes": budget.column_memory_budget_bytes,
            "segment_concurrency": budget.segment_concurrency,
        },
    )

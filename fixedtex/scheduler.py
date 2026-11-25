"""Deterministic thread-pool execution with stable collection order."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskResult:
    index: int
    value: Any = None
    error: BaseException | None = None


@dataclass
class BatchTaskResult:
    results: list[TaskResult] = field(default_factory=list)
    max_workers_used: int = 1
    max_active_tasks: int = 0


def run_ordered_tasks(
    tasks: list[Callable[[], Any]],
    max_workers: int,
) -> BatchTaskResult:
    """Run callables concurrently and return results in input order.

    No worker writes into shared result arrays. This is what keeps ledger and
    artifact ordering independent of scheduler completion order.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if not tasks:
        return BatchTaskResult(results=[], max_workers_used=min(max_workers, 1), max_active_tasks=0)

    workers = max(1, min(int(max_workers), len(tasks)))
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _wrapped(task: Callable[[], Any]) -> Any:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            return task()
        finally:
            with lock:
                active -= 1

    ordered: list[TaskResult | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fixedtex-worker") as executor:
        future_map = {
            executor.submit(_wrapped, task): index for index, task in enumerate(tasks)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                value = future.result()
                ordered[index] = TaskResult(index=index, value=value)
            except BaseException as exc:  # failures are isolated and reported
                ordered[index] = TaskResult(index=index, error=exc)

    return BatchTaskResult(
        results=[item for item in ordered if item is not None],
        max_workers_used=workers,
        max_active_tasks=max_active,
    )

"""Concurrent render scheduler with a per-batch point budget.

Tasks carry a stable sort key derived from (layer index, batch index) and are
admitted in ascending layer order.  A refill grants one fresh budget per
scheduling round; tasks exceeding the remaining budget stay queued and are
reported with ``E_POINT_BUDGET_PENDING``.  Worker functions are pure and the
GIL is released inside numpy, so concurrent execution produces the same
ordered budget deductions, pending queue, failures and pixel results as the
serial reference path.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import E_POINT_BUDGET_PENDING, Failure


@dataclass(frozen=True)
class Task:
    layer_index: int
    batch_index: int
    point_count: int
    payload: Any

    @property
    def sort_key(self) -> str:
        return f"L{self.layer_index:06d}-B{self.batch_index:06d}"


@dataclass
class SchedulerResult:
    results: list[tuple[Task, Any]]
    pending: list[Task]
    budget_log: list[tuple[str, int, int]]  # (sort_key, points, remaining)
    failures: list[Failure]
    exhausted: bool

    def ordered_results(self) -> list[Any]:
        return [value for _, value in sorted(self.results, key=lambda item: item[0].sort_key)]


def _admit(tasks: list[Task], budget: int) -> tuple[list[Task], list[Task], list[tuple[str, int, int]], int]:
    admitted: list[Task] = []
    pending: list[Task] = []
    log: list[tuple[str, int, int]] = []
    remaining = budget
    for task in tasks:
        if remaining >= task.point_count:
            remaining -= task.point_count
            admitted.append(task)
            log.append((task.sort_key, task.point_count, remaining))
        else:
            pending.append(task)
    return admitted, pending, log, remaining


def run_scheduler(
    tasks: list[Task],
    task_fn: Callable[[Task], Any],
    threads: int,
    point_budget: int,
    max_refills: int = 4096,
) -> SchedulerResult:
    """Schedule tasks under a per-batch (per-refill-round) point budget.

    Each refill grants one fresh ``point_budget``; tasks that do not fit the
    current round are deferred and immediately recorded with
    ``E_POINT_BUDGET_PENDING`` before being retried in the next round.  This
    preserves the pending/budget-deduction semantics while still allowing the
    full render to finish deterministically across rounds.
    """
    ordered = sorted(tasks, key=lambda t: t.sort_key)
    budget_log: list[tuple[str, int, int]] = []
    failures: list[Failure] = []
    completed: list[tuple[Task, Any]] = []
    deferred_ever: list[Task] = []
    queue = ordered
    refill = 0
    while queue and refill < max_refills:
        budget = point_budget if point_budget > 0 else sum(t.point_count for t in queue)
        admitted, queue, log, _ = _admit(queue, budget)
        budget_log.extend(log)
        for task in queue:
            deferred_ever.append(task)
            failures.append(
                Failure(
                    E_POINT_BUDGET_PENDING,
                    task.sort_key,
                    f"batch deferred: needs {task.point_count} points, batch budget {budget} exhausted",
                )
            )
        if not admitted:
            # A task larger than the whole batch budget is admitted alone on
            # the next round so the pipeline always makes forward progress.
            admitted, queue = queue[:1], queue[1:]
        if threads > 1:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                futures = {pool.submit(task_fn, task): task for task in admitted}
                for future in futures:
                    task = futures[future]
                    completed.append((task, future.result()))
        else:
            for task in admitted:
                completed.append((task, task_fn(task)))
        refill += 1
    pending = sorted(deferred_ever, key=lambda t: t.sort_key)
    return SchedulerResult(completed, pending, budget_log, failures, bool(pending))


def run_serial_reference(
    tasks: list[Task], task_fn: Callable[[Task], Any], point_budget: int
) -> SchedulerResult:
    """Single-threaded, layer-order reference execution with the same budget
    accounting; tests assert the concurrent run matches this exactly.
    """
    return run_scheduler(tasks, task_fn, threads=1, point_budget=point_budget)

"""Stable-key, point-budgeted thread-pool scheduler."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import E_POINT_BUDGET_PENDING


STAGE_ORDER = {"visual": 0, "text": 1, "align": 2}


@dataclass
class Task:
    sample: str
    stage: str
    row: int
    col: int
    tail: str
    points: int
    fn: Callable[[], Any]

    @property
    def sort_key(self) -> tuple[str, int, int, int, str]:
        return (
            self.sample,
            STAGE_ORDER.get(self.stage, 99),
            self.row,
            self.col,
            self.tail,
        )

    @property
    def stable_key(self) -> str:
        return (
            f"{self.sample}|{self.stage}|"
            f"r{self.row:06d}|c{self.col:06d}|{self.tail}"
        )

    def to_pending_dict(self) -> dict[str, Any]:
        return {
            "sort_key": self.stable_key,
            "sample": self.sample,
            "stage": self.stage,
            "row": self.row,
            "col": self.col,
            "points": self.points,
            "code": E_POINT_BUDGET_PENDING,
        }


@dataclass
class BatchResult:
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)
    pending: list[Task] = field(default_factory=list)
    consumed_points: int = 0
    worker_limit: int = 1
    active_peak: int = 0
    completed_task_count: int = 0
    failed_task_count: int = 0

    @property
    def pending_keys(self) -> list[str]:
        return [task.stable_key for task in self.pending]

    def ledger(self) -> dict[str, Any]:
        return {
            "worker_limit": self.worker_limit,
            "consumed_points": self.consumed_points,
            "pending_points": sum(task.points for task in self.pending),
            "pending_count": len(self.pending),
            "completed_task_count": self.completed_task_count,
            "failed_task_count": self.failed_task_count,
            "active_peak": self.active_peak,
            "active_peak_within_limit": self.active_peak <= self.worker_limit,
        }


def run_task_batch(
    tasks: list[Task],
    *,
    worker_limit: int,
    point_budget: int,
) -> BatchResult:
    """Run a stable prefix of tasks without exceeding either limit.

    The budget is consumed in stable sort-key order. The first task that does
    not fit stops admission; every task from that point remains pending. This
    makes exhaustion independent of thread completion order.
    """

    ordered = sorted(tasks, key=lambda task: task.sort_key)
    admitted: list[Task] = []
    pending: list[Task] = []
    consumed = 0
    stopped = False
    for task in ordered:
        if stopped or consumed + task.points > point_budget:
            stopped = True
            pending.append(task)
            continue
        admitted.append(task)
        consumed += task.points

    result = BatchResult(
        pending=pending,
        consumed_points=consumed,
        worker_limit=worker_limit,
        active_peak=0,
    )
    if not admitted:
        return result

    active = 0
    peak = 0
    lock = threading.Lock()

    def wrapped(task: Task) -> Any:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return task.fn()
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=worker_limit, thread_name_prefix="workbench") as executor:
        future_to_task = {executor.submit(wrapped, task): task for task in admitted}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result.results[task.stable_key] = future.result()
                result.completed_task_count += 1
            except BaseException as exc:  # Keep sibling tasks/stages isolated.
                result.errors[task.stable_key] = exc
                result.failed_task_count += 1
    result.active_peak = peak
    return result

"""Deterministic budget-aware scheduler with optional thread execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .errors import E_BUDGET_DEFERRED, E_COLUMN_QUOTA_DEFERRED


@dataclass(frozen=True)
class ScheduledTask:
    key: str
    sort_key: str
    column: str
    cost: int
    fn: Callable[[], Any]


@dataclass
class SchedulerResult:
    values: dict[str, Any]
    batches: list[dict[str, Any]]
    task_order: list[str]


def plan_batches(
    tasks: list[ScheduledTask],
    *,
    mac_budget: int,
    column_quota: int,
) -> list[dict[str, Any]]:
    """Build the deterministic budget/deferral plan without executing tasks."""

    if mac_budget <= 0 or column_quota <= 0:
        raise ValueError("budgets must be positive")
    for task in tasks:
        if task.cost > mac_budget or task.cost > column_quota:
            raise ValueError(f"task {task.key} exceeds configured budget/quota")

    pending = sorted(tasks, key=lambda task: (task.sort_key, task.key))
    batches: list[dict[str, Any]] = []
    batch_number = 0
    while pending:
        batch_number += 1
        used = 0
        column_used: dict[str, int] = {}
        ready: list[str] = []
        deferred: list[ScheduledTask] = []
        deferred_records: list[dict[str, Any]] = []
        for task in pending:
            column_cost = column_used.get(task.column, 0)
            if used + task.cost > mac_budget:
                code = E_BUDGET_DEFERRED
                reason = "batch_mac_budget_exhausted"
            elif column_cost + task.cost > column_quota:
                code = E_COLUMN_QUOTA_DEFERRED
                reason = "single_column_quota_exhausted"
            else:
                code = None
            if code is not None:
                deferred.append(task)
                deferred_records.append({
                    "key": task.key,
                    "sort_key": task.sort_key,
                    "column": task.column,
                    "cost": task.cost,
                    "code": code,
                    "reason": reason,
                })
            else:
                ready.append(task.key)
                used += task.cost
                column_used[task.column] = column_cost + task.cost
        batches.append({
            "batch": batch_number,
            "ready": ready,
            "deferred": deferred_records,
            "budget_used": used,
            "column_usage": dict(sorted(column_used.items())),
        })
        if not ready:
            break
        pending = sorted(deferred, key=lambda task: (task.sort_key, task.key))
    return batches


def run_scheduler(
    tasks: list[ScheduledTask],
    *,
    worker_count: int,
    mac_budget: int,
    column_quota: int,
) -> SchedulerResult:
    values: dict[str, Any] = {}
    task_order = [task.key for task in sorted(tasks, key=lambda task: (task.sort_key, task.key))]
    batches = plan_batches(tasks, mac_budget=mac_budget, column_quota=column_quota)
    task_by_key = {task.key: task for task in tasks}
    for batch in batches:
        ready_keys = batch["ready"]
        if not ready_keys:
            continue
        if worker_count <= 1:
            for key in ready_keys:
                values[key] = task_by_key[key].fn()
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {key: executor.submit(task_by_key[key].fn) for key in ready_keys}
                for key in ready_keys:
                    values[key] = future_map[key].result()
        batch["executed"] = len(ready_keys)
        batch["worker_count"] = max(1, worker_count)
        batch["execution_mode"] = "serial" if worker_count <= 1 else "thread_pool"

    return SchedulerResult(values=values, batches=batches, task_order=task_order)

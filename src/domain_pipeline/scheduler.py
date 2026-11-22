from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ScheduledTask:
    key: str
    cost: int
    function: Callable[[], Any]


@dataclass
class ScheduleResult:
    results: dict[str, Any]
    pending: list[str]
    budget_report: dict[str, Any]
    selected_keys: list[str]


def run_tasks(
    tasks: list[ScheduledTask],
    *,
    max_workers: int,
    budget: int | None,
    budget_name: str,
) -> ScheduleResult:
    ordered = sorted(tasks, key=lambda task: task.key)
    selected: list[ScheduledTask] = []
    pending: list[str] = []
    remaining = budget
    sequence: list[dict[str, Any]] = []
    budget_exhausted = False

    for task in ordered:
        if task.cost < 0:
            raise ValueError("task cost cannot be negative")
        if budget is None:
            selected.append(task)
            sequence.append(
                {
                    "key": task.key,
                    "cost": task.cost,
                    "remaining": None,
                }
            )
            continue
        if task.cost == 0 and not budget_exhausted:
            selected.append(task)
            sequence.append(
                {
                    "key": task.key,
                    "cost": 0,
                    "remaining": remaining,
                }
            )
            continue
        if budget_exhausted or remaining is None or remaining < task.cost:
            budget_exhausted = True
            pending.append(task.key)
            continue
        remaining -= task.cost
        selected.append(task)
        sequence.append(
            {
                "key": task.key,
                "cost": task.cost,
                "remaining": remaining,
            }
        )

    results: dict[str, Any] = {}
    if selected:
        if max_workers <= 1 or len(selected) == 1:
            for task in selected:
                results[task.key] = task.function()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(task.function) for task in selected]
                for task, future in zip(selected, futures):
                    results[task.key] = future.result()

    return ScheduleResult(
        results=results,
        pending=pending,
        budget_report={
            "name": budget_name,
            "limit": budget,
            "consumed": sum(item["cost"] for item in sequence),
            "remaining": remaining,
            "sequence": sequence,
            "pending": pending,
        },
        selected_keys=[task.key for task in selected],
    )


def parallel_ordered(
    items: list[tuple[str, Callable[[], Any]]],
    *,
    max_workers: int,
) -> dict[str, Any]:
    tasks = [ScheduledTask(key=key, cost=0, function=fn) for key, fn in items]
    return run_tasks(
        tasks,
        max_workers=max_workers,
        budget=None,
        budget_name="unbudgeted",
    ).results

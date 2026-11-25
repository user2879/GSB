"""Stable global comparison-budget scheduler with bounded concurrency."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ComparisonTask:
    stable_key: str
    stage: str
    command_word: str
    template_id: str
    cost: int
    references: tuple[str, ...] = field(default_factory=tuple)
    payload: dict[str, Any] | None = None


def schedule_tasks(
    tasks: list[ComparisonTask],
    worker: Callable[[ComparisonTask], Any],
    *,
    budget: int,
    max_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(tasks, key=lambda task: task.stable_key)
    consumed = 0
    selected: list[ComparisonTask] = []
    pending: list[dict[str, Any]] = []
    exhausted = False
    for task in ordered:
        if not exhausted and consumed + task.cost <= budget:
            selected.append(task)
            consumed += task.cost
        else:
            exhausted = True
            pending.append(
                {
                    "code": "E_BUDGET_PENDING",
                    "stable_sort_key": task.stable_key,
                    "stage": task.stage,
                    "command_word": task.command_word,
                    "template_id": task.template_id,
                    "comparison_points": task.cost,
                    "references": list(task.references),
                    "remaining_budget": max(0, budget - consumed),
                }
            )

    results: dict[str, Any] = {}
    peak_workers = 0
    active = 0
    lock = threading.Lock()

    def wrapped(task: ComparisonTask) -> tuple[str, Any]:
        nonlocal active, peak_workers
        with lock:
            active += 1
            peak_workers = max(peak_workers, active)
        try:
            return task.stable_key, worker(task)
        finally:
            with lock:
                active -= 1

    workers = max(1, min(int(max_workers), 64))
    if selected:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="voice-diag") as pool:
            futures = {pool.submit(wrapped, task): task for task in selected}
            completed = []
            for future in as_completed(futures):
                task = futures[future]
                key, value = future.result()
                completed.append((key, value))
            for key, value in sorted(completed, key=lambda item: item[0]):
                results[key] = value
    report = {
        "budget_points": budget,
        "consumed_points": consumed,
        "remaining_points": max(0, budget - consumed),
        "scheduled_task_count": len(selected),
        "pending_task_count": len(pending),
        "max_workers_configured": int(max_workers),
        "max_workers_used": workers,
        "peak_concurrency": peak_workers,
        "stable_schedule_keys": [task.stable_key for task in ordered],
    }
    return results, pending, report

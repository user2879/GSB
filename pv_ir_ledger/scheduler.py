from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


def stable_task_key(image: str, component_id: int, stage_name: str) -> str:
    return f"{image}|{int(component_id):08d}|{stage_name}"


def plan_budget(
    tasks: list[dict[str, Any]], budget: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(tasks, key=lambda task: task["scheduler_key"])
    scheduled: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    deductions: list[dict[str, Any]] = []
    remaining = int(budget)
    stopped = False
    sequence_index = 0
    for task in ordered:
        cost = int(task["cost"])
        if stopped:
            pending.append(task)
            continue
        if cost <= remaining:
            remaining -= cost
            scheduled.append(task)
            deductions.append(
                {
                    "sequence_index": sequence_index,
                    "scheduler_key": task["scheduler_key"],
                    "cost": cost,
                    "remaining_budget": remaining,
                }
            )
            sequence_index += 1
        else:
            stopped = True
            pending.append(task)
    return scheduled, pending, deductions


def run_concurrent(
    tasks: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    max_workers: int,
) -> tuple[list[Any], dict[str, Any]]:
    if not tasks:
        return [], {"max_configured_concurrency": max_workers, "max_observed_concurrency": 0}
    limit = max(1, min(int(max_workers), len(tasks)))
    active = 0
    observed = 0
    lock = threading.Lock()

    def guarded(task: dict[str, Any]) -> Any:
        nonlocal active, observed
        with lock:
            active += 1
            observed = max(observed, active)
        try:
            return worker(task)
        finally:
            with lock:
                active -= 1

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=limit, thread_name_prefix="pv-ir") as executor:
        future_map = {executor.submit(guarded, task): task["scheduler_key"] for task in tasks}
        for future in as_completed(future_map):
            key = future_map[future]
            results[key] = future.result()
    ordered_results = [results[task["scheduler_key"]] for task in sorted(tasks, key=lambda item: item["scheduler_key"])]
    return ordered_results, {"max_configured_concurrency": limit, "max_observed_concurrency": observed}

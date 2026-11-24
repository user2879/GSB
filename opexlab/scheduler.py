"""Deterministic budget planning and bounded parallel row execution."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Tuple

from .errors import E_BUDGET_EXHAUSTED, E_BUDGET_TOO_SMALL, E_LAYER_QUOTA, make_failure


def build_schedule(
    layers: Iterable[Tuple[str, int, Iterable[int]]],
    *,
    batch_element_budget: int,
    per_layer_element_quota: int,
    stage: str = "softmax",
) -> Dict[str, Any]:
    """Plan tasks in layer-name/row order without consulting thread timing."""
    remaining_budget = int(batch_element_budget)
    initial_budget = remaining_budget
    tasks: list[Dict[str, Any]] = []
    pending: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    budget_ledger: list[Dict[str, Any]] = []
    for layer_name, dim, rows in layers:
        layer_used = 0
        row_list = sorted(rows)
        for row_index in row_list:
            cost = int(dim)
            if remaining_budget < cost:
                code = E_BUDGET_TOO_SMALL if initial_budget < cost else E_BUDGET_EXHAUSTED
                pending.append(
                    {"layer": layer_name, "row": row_index, "elements": cost, "code": code}
                )
                failures.append(
                    make_failure(
                        code,
                        stage,
                        "batch element budget is smaller than the row element cost",
                        layer=layer_name,
                        row=row_index,
                        details={"required": cost, "remaining": remaining_budget},
                    )
                )
                continue
            if layer_used + cost > per_layer_element_quota:
                pending.append(
                    {
                        "layer": layer_name,
                        "row": row_index,
                        "elements": cost,
                        "code": E_LAYER_QUOTA,
                    }
                )
                failures.append(
                    make_failure(
                        E_LAYER_QUOTA,
                        stage,
                        "per-layer element quota would be exceeded; remaining layer rows aborted",
                        layer=layer_name,
                        row=row_index,
                        details={"used": layer_used, "quota": per_layer_element_quota, "next": cost},
                    )
                )
                continue
            before = remaining_budget
            remaining_budget -= cost
            layer_used += cost
            task = {
                "layer": layer_name,
                "row": row_index,
                "elements": cost,
                "budget_before": before,
                "budget_after": remaining_budget,
            }
            tasks.append(task)
            budget_ledger.append(dict(task))
        # Rows skipped by quota remain in the pending queue.  Later layers are
        # still considered, matching the serial plan exactly.
    return {
        "tasks": tasks,
        "pending_queue": pending,
        "failures": failures,
        "budget_ledger": budget_ledger,
        "initial_budget": initial_budget,
        "remaining_budget": remaining_budget,
        "consumed_budget": initial_budget - remaining_budget,
        "per_layer_element_quota": int(per_layer_element_quota),
    }


def execute_schedule(
    tasks: list[Dict[str, Any]],
    worker: Callable[[Dict[str, Any]], Any],
    *,
    max_workers: int,
) -> tuple[Dict[Tuple[str, int], Any], Dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    counter_lock = threading.Lock()
    active = 0
    peak = 0

    def wrapped(task: Dict[str, Any]) -> Any:
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        try:
            return worker(task)
        finally:
            with counter_lock:
                active -= 1

    results: Dict[Tuple[str, int], Any] = {}
    if not tasks:
        return results, {
            "requested_max_workers": max_workers,
            "peak_concurrent_tasks": 0,
            "task_count": 0,
            "submission_order": [],
        }
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="opexlab") as executor:
        futures = {executor.submit(wrapped, task): task for task in tasks}
        submission = [
            {"layer": task["layer"], "row": task["row"], "elements": task["elements"]}
            for task in tasks
        ]
        for future in as_completed(futures):
            task = futures[future]
            try:
                results[(task["layer"], task["row"])] = future.result()
            except Exception as exc:
                results[(task["layer"], task["row"])] = {
                    "_scheduler_error": f"{type(exc).__name__}: {exc}",
                }
    return results, {
        "requested_max_workers": max_workers,
        "peak_concurrent_tasks": peak,
        "task_count": len(tasks),
        "submission_order": submission,
    }

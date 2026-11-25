from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class StableTask:
    task_id: str
    sort_key: tuple[Any, ...]
    kind: str
    payload: Any = None


def sorted_tasks(tasks: Iterable[StableTask]) -> list[StableTask]:
    return sorted(tasks, key=lambda task: task.sort_key)


def run_stable(
    tasks: list[StableTask],
    handler: Callable[[StableTask], Any],
    workers: int = 1,
) -> tuple[list[Any], dict[str, Any]]:
    """Run tasks and return outputs in stable sort-key order.

    Thread completion order is deliberately recorded separately. Canonical
    pipeline state never contains completion_order.
    """
    ordered = sorted_tasks(tasks)
    scheduled_order = [task.task_id for task in ordered]
    if workers <= 1:
        results = [handler(task) for task in ordered]
        return results, {"workers": 1, "scheduled_order": scheduled_order, "completion_order": scheduled_order}

    by_id = {task.task_id: task for task in ordered}
    completed: dict[str, Any] = {}
    completion_order: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {executor.submit(handler, task): task.task_id for task in ordered}
        for future in as_completed(future_to_id):
            task_id = future_to_id[future]
            completed[task_id] = future.result()
            completion_order.append(task_id)
    results = [completed[task.task_id] for task in ordered]
    return results, {
        "workers": workers,
        "scheduled_order": scheduled_order,
        "completion_order": completion_order,
    }


def reduce_in_any_order(items: list[Any], order_seed: int = 0) -> list[Any]:
    """Test helper: emulate arbitrary completion and reduce by stable key."""
    import random

    ordered = list(items)
    random.Random(order_seed).shuffle(ordered)
    if all(isinstance(item, dict) and "sort_key" in item for item in items):
        return sorted(ordered, key=lambda item: item["sort_key"])
    return ordered

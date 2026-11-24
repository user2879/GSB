"""Thread-pool execution with stable ordering and budget accounting."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


@dataclass(frozen=True)
class Task:
    sort_key: str
    audio: str
    window: int
    stage: str
    payload: Any
    budget_cost: int = 1

    @staticmethod
    def make(audio: str, window: int, stage: str, payload: Any,
             budget_cost: int = 1) -> "Task":
        return Task(sort_key=f"{audio}|w{window:06d}|{stage}",
                    audio=audio, window=window, stage=stage,
                    payload=payload, budget_cost=budget_cost)


@dataclass
class ExecutionTrace:
    scheduled_order: List[str]
    completion_order: List[str]
    commit_order: List[str]
    budget_deduction_order: List[str]
    remaining_budget: int


def run_tasks(tasks: List[Task], worker: Callable[[Task], Any],
              concurrency: int, budget: int) -> Tuple[Dict[str, Any], ExecutionTrace]:
    """Run tasks concurrently but commit results in canonical sorted order.

    The returned dictionary is keyed by stable sort key; callers consume it in
    sorted() order, which makes output identical to serial execution.
    """
    canonical = sorted(tasks, key=lambda t: t.sort_key)
    scheduled_order = [t.sort_key for t in canonical]
    completion_order: List[str] = []
    results: Dict[str, Any] = {}

    if concurrency <= 1:
        for task in canonical:
            results[task.sort_key] = worker(task)
            completion_order.append(task.sort_key)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(worker, task): task for task in canonical}
            for future in as_completed(futures):
                task = futures[future]
                results[task.sort_key] = future.result()
                completion_order.append(task.sort_key)

    commit_order: List[str] = []
    budget_order: List[str] = []
    remaining = budget
    for task in canonical:
        commit_order.append(task.sort_key)
        budget_order.append(task.sort_key)
        remaining -= task.budget_cost
        if remaining < 0:
            raise RuntimeError(f"budget exhausted at {task.sort_key}")

    trace = ExecutionTrace(
        scheduled_order=scheduled_order,
        completion_order=completion_order,
        commit_order=commit_order,
        budget_deduction_order=budget_order,
        remaining_budget=remaining,
    )
    return results, trace


def run_serial(tasks: List[Task], worker: Callable[[Task], Any],
               budget: int) -> Tuple[Dict[str, Any], ExecutionTrace]:
    return run_tasks(tasks, worker, 1, budget)

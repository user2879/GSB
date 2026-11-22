"""Stable budget allocation and thread-pool execution helpers."""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BudgetAllocation:
    quotas: dict[str, int]
    reserved_epochs: dict[str, int]
    unstarted_models: list[str]
    total_reserved: int
    initial_budget: int


@dataclass
class TaskSpec:
    model: str
    stable_order_key: str
    function: Callable[[], Any]


@dataclass
class TaskOutcome:
    model: str
    stable_order_key: str
    value: Any = None
    exception_type: str | None = None
    exception_message: str | None = None
    traceback_text: str | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.exception_type is None


def allocate_budgets(
    ordered_models: list[tuple[str, str]],
    *,
    global_epochs: int,
    per_model_epochs: int,
) -> BudgetAllocation:
    """Allocate quotas in the caller-supplied stable key order.

    A model's full quota is reserved before launch. This makes allocation
    independent of thread completion order and of whether another model later
    converges early. Reserved-but-unused epochs are not recycled in the same run.
    """

    remaining = max(0, int(global_epochs))
    quotas: dict[str, int] = {}
    reserved: dict[str, int] = {}
    unstarted: list[str] = []
    total_reserved = 0
    for model_name, _stable_key in ordered_models:
        quota = min(max(0, int(per_model_epochs)), remaining)
        quotas[model_name] = quota
        reserved[model_name] = quota
        if quota <= 0:
            unstarted.append(model_name)
        else:
            remaining -= quota
            total_reserved += quota
    return BudgetAllocation(
        quotas=quotas,
        reserved_epochs=reserved,
        unstarted_models=unstarted,
        total_reserved=total_reserved,
        initial_budget=max(0, int(global_epochs)),
    )


def run_tasks_concurrently(
    tasks: list[TaskSpec],
    *,
    max_workers: int,
    stop_event: threading.Event | None = None,
) -> list[TaskOutcome]:
    ordered_tasks = sorted(tasks, key=lambda task: task.stable_order_key)
    if not ordered_tasks:
        return []
    worker_count = max(1, int(max_workers))
    if worker_count == 1 or len(ordered_tasks) == 1:
        outcomes = [_execute_task(task) for task in ordered_tasks]
        return outcomes

    outcomes_by_key: dict[str, TaskOutcome] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tabular-lab") as executor:
        future_map = {executor.submit(_execute_task, task): task for task in ordered_tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                outcome = future.result()
            except BaseException as exc:  # Defensive isolation for worker-level surprises.
                outcome = TaskOutcome(
                    model=task.model,
                    stable_order_key=task.stable_order_key,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                    traceback_text=traceback.format_exc(),
                    metadata={"stage": "unknown"},
                )
            outcomes_by_key[task.stable_order_key] = outcome
    return [outcomes_by_key[task.stable_order_key] for task in ordered_tasks]


def _execute_task(task: TaskSpec) -> TaskOutcome:
    started = time.perf_counter()
    try:
        value = task.function()
        return TaskOutcome(
            model=task.model,
            stable_order_key=task.stable_order_key,
            value=value,
            elapsed_seconds=time.perf_counter() - started,
        )
    except BaseException as exc:
        return TaskOutcome(
            model=task.model,
            stable_order_key=task.stable_order_key,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started,
            metadata={"stage": "unknown"},
        )

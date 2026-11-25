"""Budget-constrained stable scheduling for calibration and quantization."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .errors import E_BUDGET_PENDING, E_QUOTA_EXCEEDED, FailureList


@dataclass(frozen=True)
class Task:
    layer_name: str
    group_index: int
    cost: int
    payload: Any

    @property
    def sort_key(self) -> tuple[str, int]:
        return (self.layer_name, self.group_index)


@dataclass
class ScheduleResult:
    completed: list[Task]
    pending: list[Task]
    quota_aborted: list[str]
    budget_pending: list[Task]
    budget_consumed: int


def plan_tasks(tasks: list[Task]) -> list[Task]:
    # Stable order: layer name lexicographic, then group index.
    return sorted(tasks, key=lambda t: t.sort_key)


def schedule(tasks: list[Task], global_budget: int,
             per_layer_quotas: dict[str, int],
             failures: FailureList) -> ScheduleResult:
    ordered = plan_tasks(tasks)
    layer_spend: dict[str, int] = {}
    completed: list[Task] = []
    pending: list[Task] = []
    quota_aborted: set[str] = set()
    consumed = 0
    for task in ordered:
        quota = per_layer_quotas.get(task.layer_name)
        if quota is not None and layer_spend.get(task.layer_name, 0) + task.cost > quota:
            if task.layer_name not in quota_aborted:
                quota_aborted.add(task.layer_name)
                failures.add(E_QUOTA_EXCEEDED, "scheduler", task.layer_name,
                             "per-layer calibration quota exceeded",
                             f"quota={quota}")
            pending.append(task)
            failures.add(E_BUDGET_PENDING, "scheduler",
                         f"{task.layer_name}:g{task.group_index}",
                         "task held back after quota abort")
            continue
        if consumed + task.cost > global_budget:
            pending.append(task)
            failures.add(E_BUDGET_PENDING, "scheduler",
                         f"{task.layer_name}:g{task.group_index}",
                         "global calibration budget exhausted",
                         f"consumed={consumed} budget={global_budget}")
            continue
        completed.append(task)
        layer_spend[task.layer_name] = layer_spend.get(task.layer_name, 0) + task.cost
        consumed += task.cost
    return ScheduleResult(
        completed=completed,
        pending=pending,
        quota_aborted=sorted(quota_aborted),
        budget_pending=[t for t in pending if t.layer_name not in quota_aborted],
        budget_consumed=consumed,
    )


def run_map(func: Callable[[Task], Any], tasks: list[Task],
            workers: int) -> list[Any]:
    """Run tasks while guaranteeing serial/parallel result parity.

    Every task carries its stable (layer, group) key; results are always
    returned in that key order regardless of execution mode.
    """
    ordered = plan_tasks(tasks)
    if workers <= 1 or len(ordered) <= 1:
        return [func(task) for task in ordered]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, task) for task in ordered]
        results = [future.result() for future in futures]
    return results


def batch_quant_budget(planned_elements: int, batch_points: int) -> int:
    """Number of quantization batches needed under the per-batch budget."""
    if batch_points <= 0:
        raise ValueError("batch_quant_points must be positive")
    return (planned_elements + batch_points - 1) // batch_points

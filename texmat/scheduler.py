"""Deterministic task scheduling with a per-batch point budget.

Admission, budget deduction, ordering keys and output assembly are all driven by
stable integer keys, so thread scheduling timing cannot change the observable
order of pixels, reuse lists, budget events or failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import Failure


@dataclass(frozen=True)
class Task:
    key: str                 # globally stable sorting key
    group: int                # 0 = layer phase, 1 = channel/final phase
    sequence: int             # stable position within phase
    points: int               # pixel-point cost
    node_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetEvent:
    order_key: str
    node_id: str
    points: int
    remaining: int
    action: str               # admitted | pending

    def to_dict(self) -> dict[str, Any]:
        return {"order_key": self.order_key, "node_id": self.node_id,
                "points": self.points, "remaining": self.remaining, "action": self.action}


class Scheduler:
    def __init__(self, point_budget: int):
        self.point_budget = point_budget
        self.remaining = point_budget
        self.events: list[BudgetEvent] = []
        self.pending: list[Task] = []
        self.admitted: list[Task] = []

    def admit(self, tasks: list[Task]) -> list[Task]:
        """Split tasks into admitted vs pending using stable key order only.

        A single task whose cost exceeds the whole budget is admitted on a
        guaranteed dedicated batch (recorded with remaining clamped to zero) so
        progress can never deadlock; the event log preserves the order.
        """
        ordered = sorted(tasks, key=lambda task: (task.group, task.sequence, task.key))
        for task in ordered:
            if task.points <= self.remaining:
                self.remaining -= task.points
                self.events.append(BudgetEvent(task.key, task.node_id, task.points,
                                               self.remaining, "admitted"))
                self.admitted.append(task)
            elif not self.admitted and task.points > self.point_budget:
                self.remaining = 0
                self.events.append(BudgetEvent(task.key, task.node_id, task.points,
                                               self.remaining, "admitted_oversize"))
                self.admitted.append(task)
            else:
                self.events.append(BudgetEvent(task.key, task.node_id, task.points,
                                               self.remaining, "pending"))
                self.pending.append(task)
        return list(self.admitted)

    def pending_failures(self) -> list[Failure]:
        return [
            Failure(
                code="E_POINT_BUDGET_PENDING", target=task.node_id,
                message=f"retained for a later run; needs {task.points} points, "
                        f"{self.remaining} remaining",
                order_key=task.key,
                details={"points": task.points, "remaining_budget": self.remaining},
            )
            for task in sorted(self.pending, key=lambda item: (item.group, item.sequence, item.key))
        ]

    def deduction_sequence(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


def stable_layer_key(index: int) -> str:
    return f"0:layer:{index:04d}"


def stable_channel_key(material_order: int, channel_order: int) -> str:
    return f"1:channel:{material_order:04d}:{channel_order:04d}"


def stable_final_key(material_order: int) -> str:
    return f"2:final:{material_order:04d}"


def run_in_key_order(tasks: list[Task], workers: int,
                     execute: Callable[[Task], Any]) -> dict[str, Any]:
    """Execute admitted tasks with a thread pool but return results key-sorted."""
    ordered = sorted(tasks, key=lambda task: (task.group, task.sequence, task.key))
    results: dict[str, Any] = {}
    if workers <= 1 or len(ordered) <= 1:
        for task in ordered:
            results[task.key] = execute(task)
        return results
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, task): task.key for task in ordered}
        for future in futures:
            results[futures[future]] = future.result()
    return {key: results[key] for key in sorted(results)}

"""Stable single-budget scheduling with explicit pending snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import E_CALIB_BUDGET_PENDING


@dataclass(frozen=True)
class BudgetTask:
    group_id: str
    column: str
    bit_width: int
    cost: int

    @property
    def task_key(self) -> str:
        return f"{self.group_id}|{self.column}|{self.bit_width:03d}"

    @property
    def sort_key(self) -> tuple[str, str, int]:
        return (self.group_id, self.column, self.bit_width)


@dataclass(frozen=True)
class ScheduleResult:
    cycles: tuple[dict, ...]
    pending: tuple[BudgetTask, ...]
    completed: tuple[BudgetTask, ...]
    consumed: int

    @property
    def complete(self) -> bool:
        return not self.pending


def make_tasks(
    *,
    groups_to_compute: Iterable[str],
    column_groups: dict[str, tuple[str, ...]],
    bit_widths: Iterable[int],
    counts: dict[tuple[str, str], int],
) -> list[BudgetTask]:
    tasks: list[BudgetTask] = []
    widths = tuple(sorted(set(int(width) for width in bit_widths)))
    for group_id in sorted(set(groups_to_compute)):
        for column in sorted(column_groups[group_id]):
            for bit_width in widths:
                tasks.append(
                    BudgetTask(
                        group_id=group_id,
                        column=column,
                        bit_width=bit_width,
                        cost=max(0, int(counts.get((group_id, column), 0))),
                    )
                )
    return sorted(tasks, key=lambda task: task.sort_key)


def schedule_budget(
    tasks: Iterable[BudgetTask],
    *,
    budget_per_cycle: int,
    max_cycles: int | None = None,
) -> ScheduleResult:
    """Consume budget in stable group/column/width order.

    A task is never skipped while it is the queue head. Oversized tasks are split
    across cycles, and each cycle containing unfinished work emits the stable
    E_CALIB_BUDGET_PENDING marker and a full pending snapshot.
    """

    if budget_per_cycle <= 0:
        raise ValueError("budget_per_cycle must be positive")
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be positive when supplied")

    queue = sorted(list(tasks), key=lambda task: task.sort_key)
    remaining = {task.task_key: max(0, task.cost) for task in queue}
    completed: list[BudgetTask] = []
    cycles: list[dict] = []
    total_consumed = 0

    while queue:
        if max_cycles is not None and len(cycles) >= max_cycles:
            break
        budget_left = budget_per_cycle
        cycle_consumed = 0
        deductions: list[dict] = []
        cycle_completed: list[BudgetTask] = []
        while queue and budget_left > 0:
            task = queue[0]
            left_for_task = remaining[task.task_key]
            if left_for_task == 0:
                queue.pop(0)
                completed.append(task)
                cycle_completed.append(task)
                continue
            consumed = min(left_for_task, budget_left)
            remaining[task.task_key] -= consumed
            budget_left -= consumed
            cycle_consumed += consumed
            total_consumed += consumed
            deductions.append(
                {
                    "task_key": task.task_key,
                    "group_id": task.group_id,
                    "column": task.column,
                    "bit_width": task.bit_width,
                    "requested": left_for_task,
                    "consumed": consumed,
                    "remaining_after": remaining[task.task_key],
                }
            )
            if remaining[task.task_key] == 0:
                queue.pop(0)
                completed.append(task)
                cycle_completed.append(task)
            else:
                break
        pending = [task for task in queue if remaining[task.task_key] > 0]
        cycle = {
            "cycle_index": len(cycles) + 1,
            "budget_limit": budget_per_cycle,
            "consumed": cycle_consumed,
            "deductions": deductions,
            "completed_task_keys": [task.task_key for task in cycle_completed],
            "pending": [
                {
                    "task_key": task.task_key,
                    "group_id": task.group_id,
                    "column": task.column,
                    "bit_width": task.bit_width,
                    "remaining": remaining[task.task_key],
                }
                for task in pending
            ],
            "pending_count": len(pending),
            "event_code": E_CALIB_BUDGET_PENDING if pending else None,
        }
        cycles.append(cycle)
        if not queue:
            break

    pending_tasks = tuple(task for task in queue if remaining[task.task_key] > 0)
    completed_tasks = tuple(sorted(completed, key=lambda task: task.sort_key))
    return ScheduleResult(
        cycles=tuple(cycles),
        pending=pending_tasks,
        completed=completed_tasks,
        consumed=total_consumed,
    )

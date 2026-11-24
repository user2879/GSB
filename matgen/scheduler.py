"""Stable point-budget scheduling with retained pending work."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .errors import E_POINT_BUDGET_PENDING


@dataclass(frozen=True)
class WorkItem:
    stage: str
    stage_rank: int
    layer_index: int
    row: int
    col_start: int
    col_end: int
    width: int
    payload: dict[str, Any] | None = None
    row_end: int | None = None

    @property
    def row_stop(self) -> int:
        return self.row_end if self.row_end is not None else self.row + 1

    @property
    def point_count(self) -> int:
        return max(0, self.row_stop - self.row) * max(0, self.col_end - self.col_start)

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.stage_rank, self.layer_index, self.row, self.col_start)

    def deduction_key(self) -> list[int | str]:
        return [self.stage, self.stage_rank, self.layer_index, self.row, self.row_stop, self.col_start, self.col_end]

    def split(self, points: int) -> tuple["WorkItem", "WorkItem"]:
        if points <= 0 or points >= self.point_count:
            raise ValueError("split point must be inside the work item")
        if points >= self.width and self.row_stop - self.row > 1:
            rows = points // self.width
            rows = max(1, min(rows, self.row_stop - self.row - 1))
            cut = self.row + rows
            return replace(self, row_end=cut), replace(self, row=cut)
        cut = self.col_start + points
        return replace(self, col_end=cut), replace(self, col_start=cut)


class PointBudgetScheduler:
    def __init__(self, point_budget: int) -> None:
        if point_budget <= 0:
            raise ValueError("point_budget must be positive")
        self.point_budget = point_budget
        self.pending: list[WorkItem] = []
        self.events: list[dict[str, Any]] = []
        self.deduction_sequence: list[list[int | str]] = []
        self.total_consumed = 0
        self.batch_number = 0

    def add(self, items: list[WorkItem]) -> None:
        self.pending.extend(items)
        self.pending.sort(key=lambda item: item.sort_key)

    @property
    def pending_points(self) -> int:
        return sum(item.point_count for item in self.pending)

    def take_batch(self) -> list[WorkItem]:
        self.batch_number += 1
        remaining = self.point_budget
        selected: list[WorkItem] = []
        while self.pending and remaining > 0:
            item = self.pending[0]
            if item.point_count <= remaining:
                selected.append(self.pending.pop(0))
                remaining -= item.point_count
            elif item.row_stop - item.row == 1:
                prefix, suffix = item.split(remaining)
                selected.append(prefix)
                self.pending[0] = suffix
                remaining = 0
            else:
                # Keep a multi-row item intact and carry the unused budget forward.
                break
        consumed = sum(item.point_count for item in selected)
        self.total_consumed += consumed
        self.deduction_sequence.extend(item.deduction_key() for item in selected)
        pending_points = self.pending_points
        event = {
            "batch": self.batch_number,
            "budget": self.point_budget,
            "consumed": consumed,
            "pending_points": pending_points,
            "pending_count": len(self.pending),
            "deduction_sequence": [item.deduction_key() for item in selected],
            "status": E_POINT_BUDGET_PENDING if pending_points else "complete",
        }
        if pending_points:
            event["code"] = E_POINT_BUDGET_PENDING
        self.events.append(event)
        return selected

    def drain(self) -> list[list[WorkItem]]:
        batches: list[list[WorkItem]] = []
        while self.pending:
            selected = self.take_batch()
            if not selected:
                break
            batches.append(selected)
        return batches

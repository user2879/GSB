"""Deterministic budget planning independent of worker completion order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class BudgetPlan:
    limit: int
    consumed: int
    remaining: int
    ledger: list[dict[str, Any]] = field(default_factory=list)
    admitted_pages: list[str] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    closed_early: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "entries": self.ledger,
            "admitted_pages": self.admitted_pages,
            "pending": self.pending,
            "closed_early": self.closed_early,
        }


def _make_entry(
    sequence: int,
    file_name: str,
    points: int,
    remaining_before: int,
    remaining_after: int,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "file": file_name,
        "sort_key": f"{file_name}|line",
        "points": points,
        "remaining_before": remaining_before,
        "remaining_after": remaining_after,
    }


def plan_budget(
    ordered_files: Sequence[str],
    line_costs: Mapping[str, int],
    limit: int,
    *,
    prior_plan: Mapping[str, Any] | None = None,
) -> BudgetPlan:
    """Plan line-detection admissions in stable filename order.

    Once a positive-cost page cannot be admitted, later line detections are
    queued rather than allowing small later pages to overtake the stable queue.
    """

    if prior_plan is not None:
        ledger = [dict(entry) for entry in prior_plan.get("entries", [])]
        admitted_pages = [str(value) for value in prior_plan.get("admitted_pages", [])]
        admitted_set = set(admitted_pages)
        consumed = sum(int(entry["points"]) for entry in ledger)
        remaining = max(0, limit - consumed)
        pending = []
        for file_name in ordered_files:
            if file_name not in line_costs or file_name in admitted_set:
                continue
            pending.append(
                {
                    "file": file_name,
                    "sort_key": f"{file_name}|line",
                    "code": "E_POINT_BUDGET_PENDING",
                    "points_required": int(line_costs[file_name]),
                }
            )
        return BudgetPlan(
            limit=limit,
            consumed=consumed,
            remaining=remaining,
            ledger=sorted(ledger, key=lambda entry: int(entry["sequence"])),
            admitted_pages=sorted(admitted_pages),
            pending=pending,
            closed_early=bool(prior_plan.get("closed_early", False)),
        )

    ledger: list[dict[str, Any]] = []
    admitted_pages: list[str] = []
    pending: list[dict[str, Any]] = []
    remaining = limit
    closed = False
    sequence = 1
    for file_name in ordered_files:
        if file_name not in line_costs:
            continue
        points = int(line_costs[file_name])
        if closed or (points > 0 and points > remaining):
            pending.append(
                {
                    "file": file_name,
                    "sort_key": f"{file_name}|line",
                    "code": "E_POINT_BUDGET_PENDING",
                    "points_required": points,
                }
            )
            if points > 0:
                closed = True
            continue
        before = remaining
        remaining -= points
        admitted_pages.append(file_name)
        ledger.append(_make_entry(sequence, file_name, points, before, remaining))
        sequence += 1

    return BudgetPlan(
        limit=limit,
        consumed=limit - remaining,
        remaining=remaining,
        ledger=ledger,
        admitted_pages=sorted(admitted_pages),
        pending=pending,
        closed_early=closed,
    )

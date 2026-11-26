from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar
from concurrent.futures import ThreadPoolExecutor

from .codes import E_POINT_BUDGET_PENDING, E_QUOTA_EXCEEDED

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BudgetPlan:
    candidate_ids: list[str]
    approved_ids: list[str] = field(default_factory=list)
    pending_ids: list[str] = field(default_factory=list)
    quota_blocked_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    consumed_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": self.candidate_ids,
            "approved_ids": self.approved_ids,
            "pending_ids": self.pending_ids,
            "quota_blocked_ids": self.quota_blocked_ids,
            "events": self.events,
            "consumed_points": self.consumed_points,
        }


def plan_point_budget(
    units: list[dict[str, Any]],
    pending_ids: list[str],
    global_point_budget: int,
    column_point_quota: int,
) -> BudgetPlan:
    by_id = {unit["unit_id"]: unit for unit in units}
    priority = [unit_id for unit_id in pending_ids if unit_id in by_id]
    stable = [unit["unit_id"] for unit in units]
    candidates = priority + [unit_id for unit_id in stable if unit_id not in priority]
    plan = BudgetPlan(candidate_ids=candidates)
    budget_open = True
    consumed = 0

    for unit_id in candidates:
        unit = by_id[unit_id]
        points = int(unit["planned_points"])
        if points > column_point_quota:
            plan.quota_blocked_ids.append(unit_id)
            plan.events.append(
                {
                    "code": E_QUOTA_EXCEEDED,
                    "target": unit_id,
                    "file_name": unit["image_id"] + ".png",
                    "column_rank": unit["column_rank"],
                    "planned_points": points,
                    "column_point_quota": column_point_quota,
                    "stable_order": unit["stable_order"],
                }
            )
            continue

        if not budget_open or consumed + points > global_point_budget:
            budget_open = False
            if unit_id not in plan.pending_ids:
                plan.pending_ids.append(unit_id)
            plan.events.append(
                {
                    "code": E_POINT_BUDGET_PENDING,
                    "target": unit_id,
                    "file_name": unit["image_id"] + ".png",
                    "column_rank": unit["column_rank"],
                    "planned_points": points,
                    "global_point_budget": global_point_budget,
                    "consumed_points": consumed,
                    "priority_from_previous_batch": unit_id in priority,
                }
            )
            continue

        plan.approved_ids.append(unit_id)
        consumed += points

    plan.consumed_points = consumed
    return plan


def allocate_comparisons(
    prepared_order: list[dict[str, Any]], max_component_comparisons: int
) -> tuple[list[str], list[dict[str, Any]], int]:
    approved: list[str] = []
    events: list[dict[str, Any]] = []
    used = 0
    budget_open = True
    for prepared in prepared_order:
        unit_id = prepared["unit_id"]
        demand = int(prepared["comparison_demand"])
        if not budget_open or used + demand > max_component_comparisons:
            budget_open = False
            events.append(
                {
                    "code": "E_COMPARISON_LIMIT",
                    "target": unit_id,
                    "demand": demand,
                    "used": used,
                    "limit": max_component_comparisons,
                }
            )
            continue
        approved.append(unit_id)
        used += demand
    return approved, events, used


def run_ordered(
    items: list[T],
    worker: Callable[[T], R],
    max_workers: int,
) -> list[R]:
    """Run worker concurrently but return results in caller's stable item order."""
    if max_workers <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        return [future.result() for future in futures]

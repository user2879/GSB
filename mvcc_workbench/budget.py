from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json
from .model import Dataset, Query
from .visibility import estimate_query_cost


@dataclass(frozen=True)
class Allocation:
    query_id: str
    read_point_id: str
    row_key: str
    priority: int
    estimated_cost: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "priority": self.priority,
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class DeferredQuery:
    query_id: str
    read_point_id: str
    row_key: str
    priority: int
    estimated_cost: int
    reason: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "priority": self.priority,
            "estimated_cost": self.estimated_cost,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BudgetRound:
    round_no: int
    round_cost_budget: int
    allocated_cost: int
    allocations: tuple[Allocation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_no": self.round_no,
            "round_cost_budget": self.round_cost_budget,
            "allocated_cost": self.allocated_cost,
            "allocations": [item.as_dict() for item in self.allocations],
        }


@dataclass(frozen=True)
class BudgetPlan:
    global_cost_budget: int
    round_cost_budget: int
    rounds: tuple[BudgetRound, ...]
    deferred: tuple[DeferredQuery, ...]
    total_allocated_cost: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_cost_budget": self.global_cost_budget,
            "round_cost_budget": self.round_cost_budget,
            "total_allocated_cost": self.total_allocated_cost,
            "rounds": [item.as_dict() for item in self.rounds],
            "deferred": [item.as_dict() for item in self.deferred],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())

    def allocated_query_ids(self) -> tuple[str, ...]:
        return tuple(
            allocation.query_id
            for budget_round in self.rounds
            for allocation in budget_round.allocations
        )


def build_budget_plan(dataset: Dataset) -> BudgetPlan:
    costs = {query.query_id: estimate_query_cost(dataset, query) for query in dataset.queries}
    pending: list[Query] = list(dataset.sorted_queries(priority_first=True))
    remaining_global = dataset.budget.global_cost_budget
    rounds: list[BudgetRound] = []
    deferred: list[DeferredQuery] = []
    round_no = 1

    while pending and remaining_global > 0:
        round_cap = min(dataset.budget.round_cost_budget, remaining_global)
        allocations: list[Allocation] = []
        used = 0
        made_progress = False
        while pending:
            query = pending[0]
            cost = costs[query.query_id]
            if cost > round_cap:
                if not allocations and cost > dataset.budget.round_cost_budget:
                    query = pending.pop(0)
                    deferred.append(
                        DeferredQuery(
                            query_id=query.query_id,
                            read_point_id=query.read_point_id,
                            row_key=query.row_key,
                            priority=query.priority,
                            estimated_cost=cost,
                            reason="COST_EXCEEDS_ROUND_BUDGET",
                            detail="single-query estimate cannot fit inside one round budget",
                        )
                    )
                    made_progress = True
                    continue
                break
            if used + cost > round_cap:
                break
            pending.pop(0)
            allocations.append(
                Allocation(
                    query_id=query.query_id,
                    read_point_id=query.read_point_id,
                    row_key=query.row_key,
                    priority=query.priority,
                    estimated_cost=cost,
                )
            )
            used += cost
            remaining_global -= cost
            made_progress = True
        if allocated_cost := used:
            rounds.append(
                BudgetRound(
                    round_no=round_no,
                    round_cost_budget=round_cap,
                    allocated_cost=allocated_cost,
                    allocations=tuple(allocations),
                )
            )
            round_no += 1
        if not made_progress:
            break

    for query in pending:
        cost = costs[query.query_id]
        if cost > remaining_global:
            reason = "GLOBAL_BUDGET_EXHAUSTED"
            detail = "remaining global cost budget is smaller than the next stable-priority query"
        else:
            reason = "ROUND_PACKING_STOPPED"
            detail = "deterministic stable-priority packing stopped to avoid exceeding a budget"
        deferred.append(
            DeferredQuery(
                query_id=query.query_id,
                read_point_id=query.read_point_id,
                row_key=query.row_key,
                priority=query.priority,
                estimated_cost=cost,
                reason=reason,
                detail=detail,
            )
        )

    total_allocated = sum(item.allocated_cost for item in rounds)
    if total_allocated > dataset.budget.global_cost_budget:
        raise AssertionError("budget plan exceeded global budget")
    return BudgetPlan(
        global_cost_budget=dataset.budget.global_cost_budget,
        round_cost_budget=dataset.budget.round_cost_budget,
        rounds=tuple(rounds),
        deferred=tuple(deferred),
        total_allocated_cost=total_allocated,
    )

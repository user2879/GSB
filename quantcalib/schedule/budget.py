"""Global / per-layer / per-batch point budget scheduler.

Planning is fully deterministic: layers are consumed in lexicographic order
and groups by group index. The plan never depends on which thread finishes
first; a bounded thread pool only executes already-planned units.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Callable

from quantcalib.errors import E_POINT_BUDGET_PENDING, E_QUOTA_EXCEEDED


@dataclass
class WorkUnit:
    layer: str
    group_index: int
    point_cost: int
    stage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScheduleOutcome:
    planned: list[WorkUnit]
    pending: list[WorkUnit]
    failures: list[dict[str, Any]]
    consumed_budget: int
    per_layer_consumed: dict[str, int]
    max_workers_observed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned": [u.to_dict() for u in self.planned],
            "pending": [u.to_dict() for u in self.pending],
            "failures": self.failures,
            "consumed_budget": self.consumed_budget,
            "per_layer_consumed": self.per_layer_consumed,
            "max_workers_observed": self.max_workers_observed,
        }


class BudgetScheduler:
    def __init__(self, max_workers: int, global_budget: int,
                 per_layer_quota: int, batch_budget: int) -> None:
        self.max_workers = max_workers
        self.global_budget = global_budget
        self.per_layer_quota = per_layer_quota
        self.batch_budget = batch_budget

    def plan(self, units: list[WorkUnit],
             already_consumed: int = 0,
             per_layer_consumed: dict[str, int] | None = None
             ) -> tuple[list[WorkUnit], list[WorkUnit], list[dict[str, Any]]]:
        ordered = sorted(units, key=lambda u: (u.layer, u.group_index, u.stage))
        layer_used = dict(per_layer_consumed or {})
        global_used = already_consumed
        planned: list[WorkUnit] = []
        pending: list[WorkUnit] = []
        failures: list[dict[str, Any]] = []
        batch_used = 0

        for unit in ordered:
            layer_total = layer_used.get(unit.layer, 0)
            if layer_total + unit.point_cost > self.per_layer_quota:
                failures.append({
                    "code": E_QUOTA_EXCEEDED,
                    "layer": unit.layer,
                    "group_index": unit.group_index,
                    "detail": {
                        "cost": unit.point_cost,
                        "layer_consumed": layer_total,
                        "quota": self.per_layer_quota,
                    },
                })
                continue
            if global_used + unit.point_cost > self.global_budget:
                pending.append(unit)
                failures.append({
                    "code": E_POINT_BUDGET_PENDING,
                    "layer": unit.layer,
                    "group_index": unit.group_index,
                    "detail": {"cost": unit.point_cost,
                               "global_consumed": global_used,
                               "global_budget": self.global_budget},
                })
                continue
            if batch_used + unit.point_cost > self.batch_budget:
                # New batch boundary; pending remains empty because the unit
                # still fits globally, it just starts the next batch.
                batch_used = 0
            planned.append(unit)
            layer_used[unit.layer] = layer_total + unit.point_cost
            global_used += unit.point_cost
            batch_used += unit.point_cost
        return planned, pending, failures

    def run(self, units: list[WorkUnit], worker: Callable[[WorkUnit], Any],
            already_consumed: int = 0,
            per_layer_consumed: dict[str, int] | None = None
            ) -> tuple[ScheduleOutcome, list[Any]]:
        planned, pending, failures = self.plan(
            units, already_consumed, per_layer_consumed)
        results: list[Any] = []
        observed = 0
        if planned:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                observed = min(self.max_workers, len(planned))
                futures = [pool.submit(worker, unit) for unit in planned]
                # Collect in planned order, not completion order.
                results = [future.result() for future in futures]
        consumed = already_consumed + sum(u.point_cost for u in planned)
        layer_used = dict(per_layer_consumed or {})
        for unit in planned:
            layer_used[unit.layer] = layer_used.get(unit.layer, 0) + unit.point_cost
        outcome = ScheduleOutcome(
            planned=planned, pending=pending, failures=failures,
            consumed_budget=consumed, per_layer_consumed=layer_used,
            max_workers_observed=observed,
        )
        return outcome, results

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .models import AttentionSample, PositionConfig


E_SAMPLE_BUDGET_EXHAUSTED = "E_SAMPLE_BUDGET_EXHAUSTED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"


@dataclass(frozen=True)
class WorkTask:
    sample: AttentionSample
    scheme: str
    quota: int
    required_points: int
    status: str
    error_code: str | None
    charged_points: int
    sequence: int | None

    @property
    def identity(self) -> tuple[str, str]:
        return (self.sample.sample_id, self.scheme)

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.identity

    @property
    def scheduled(self) -> bool:
        return self.status == "scheduled"


@dataclass(frozen=True)
class BudgetPlan:
    global_point_capacity: int
    global_mac_budget: int
    point_mac_cost: int
    sample_quota: int
    tasks: tuple[WorkTask, ...]
    deduction_sequence: tuple[dict[str, Any], ...]
    remaining_points: int
    consumed_points: int
    exhausted_tasks: tuple[WorkTask, ...]
    pending_queue: tuple[WorkTask, ...]

    @property
    def scheduled_tasks(self) -> tuple[WorkTask, ...]:
        return tuple(task for task in self.tasks if task.scheduled)

    @property
    def consumed_mac(self) -> int:
        return self.consumed_points * self.point_mac_cost

    @property
    def remaining_mac(self) -> int:
        return max(0, self.global_mac_budget - self.consumed_mac)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_point_quota": self.sample_quota,
            "global_mac_budget": self.global_mac_budget,
            "global_point_capacity": self.global_point_capacity,
            "point_mac_cost": self.point_mac_cost,
            "consumed_points": self.consumed_points,
            "consumed_mac": self.consumed_mac,
            "remaining_points": self.remaining_points,
            "remaining_mac": self.remaining_mac,
            "scheduled_task_count": len(self.scheduled_tasks),
            "deduction_sequence": list(self.deduction_sequence),
            "budget_exhausted_samples": [
                {
                    "sample_id": task.sample.sample_id,
                    "scheme": task.scheme,
                    "required_points": task.required_points,
                    "sample_quota": task.quota,
                    "error_code": task.error_code,
                }
                for task in self.exhausted_tasks
            ],
            "pending_queue": [
                {
                    "sample_id": task.sample.sample_id,
                    "scheme": task.scheme,
                    "required_points": task.required_points,
                    "error_code": task.error_code,
                }
                for task in self.pending_queue
            ],
        }


def build_budget_plan(config: PositionConfig, samples: Sequence[AttentionSample]) -> BudgetPlan:
    point_mac_cost = max(1, config.dimension)
    global_point_capacity = config.global_mac_budget // point_mac_cost
    remaining = global_point_capacity
    sequence = 0
    global_exhausted = False
    tasks: list[WorkTask] = []
    deductions: list[dict[str, Any]] = []

    for sample in sorted(samples, key=lambda item: item.sample_id):
        for scheme in sorted(config.active_schemes):
            required = sample.point_units
            quota = config.batch_point_budget
            if global_exhausted:
                tasks.append(
                    WorkTask(
                        sample=sample,
                        scheme=scheme,
                        quota=quota,
                        required_points=required,
                        status="budget_pending",
                        error_code=E_POINT_BUDGET_PENDING,
                        charged_points=0,
                        sequence=None,
                    )
                )
                continue

            quota_exceeded = required > quota
            points_needed_before_abort = quota if quota_exceeded else required
            if remaining < points_needed_before_abort:
                global_exhausted = True
                tasks.append(
                    WorkTask(
                        sample=sample,
                        scheme=scheme,
                        quota=quota,
                        required_points=required,
                        status="budget_pending",
                        error_code=E_POINT_BUDGET_PENDING,
                        charged_points=0,
                        sequence=None,
                    )
                )
                continue

            before = remaining
            remaining -= points_needed_before_abort
            sequence += 1
            status = "budget_exhausted" if quota_exceeded else "scheduled"
            error_code = E_SAMPLE_BUDGET_EXHAUSTED if quota_exceeded else None
            task = WorkTask(
                sample=sample,
                scheme=scheme,
                quota=quota,
                required_points=required,
                status=status,
                error_code=error_code,
                charged_points=points_needed_before_abort,
                sequence=sequence,
            )
            tasks.append(task)
            deductions.append(
                {
                    "sequence": sequence,
                    "sample_id": sample.sample_id,
                    "scheme": scheme,
                    "status": status,
                    "requested_points": required,
                    "sample_quota": quota,
                    "charged_points": points_needed_before_abort,
                    "before_points": before,
                    "after_points": remaining,
                }
            )

    ordered_tasks = tuple(tasks)
    return BudgetPlan(
        global_point_capacity=global_point_capacity,
        global_mac_budget=config.global_mac_budget,
        point_mac_cost=point_mac_cost,
        sample_quota=config.batch_point_budget,
        tasks=ordered_tasks,
        deduction_sequence=tuple(deductions),
        remaining_points=remaining,
        consumed_points=global_point_capacity - remaining,
        exhausted_tasks=tuple(task for task in ordered_tasks if task.status == "budget_exhausted"),
        pending_queue=tuple(task for task in ordered_tasks if task.status == "budget_pending"),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .errors import ErrorCode, Failure
from .samples import AttentionSample


@dataclass(frozen=True)
class SampleTask:
    scheme: str
    sample: AttentionSample
    position_key: str
    scoring_key: str
    position_hit: bool
    scoring_hit: bool
    force_execution: bool
    sort_key: str
    stage_sort_keys: dict[str, str]

    @property
    def points(self) -> int:
        return self.sample.point_count

    @property
    def cache_reusable(self) -> bool:
        return self.position_hit and self.scoring_hit and not self.force_execution


@dataclass(frozen=True)
class TaskDecision:
    task: SampleTask
    action: str
    points: int = 0
    remaining_before: int = 0
    remaining_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.task.scheme,
            "sample_id": self.task.sample.sample_id,
            "sort_key": self.task.sort_key,
            "stage_sort_keys": self.task.stage_sort_keys,
            "position_cache_key": self.task.position_key,
            "scoring_cache_key": self.task.scoring_key,
            "position_cache_hit": self.task.position_hit,
            "scoring_cache_hit": self.task.scoring_hit,
            "force_execution": self.task.force_execution,
            "action": self.action,
            "points": self.points,
            "remaining_before": self.remaining_before,
            "remaining_after": self.remaining_after,
        }


@dataclass(frozen=True)
class ScheduleResult:
    ordered_tasks: tuple[SampleTask, ...]
    decisions: tuple[TaskDecision, ...]
    approved: tuple[SampleTask, ...]
    pending: tuple[SampleTask, ...]
    reused: tuple[SampleTask, ...]
    budget_deductions: tuple[dict[str, Any], ...]
    pending_failures: tuple[Failure, ...]
    remaining_budget: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "order_rule": "ascending (scheme_name, sample_id)",
            "ordered_sort_keys": [task.sort_key for task in self.ordered_tasks],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "approved_count": len(self.approved),
            "reused_count": len(self.reused),
            "pending_count": len(self.pending),
            "budget_deductions": list(self.budget_deductions),
            "remaining_budget": self.remaining_budget,
        }


def schedule_sample_tasks(
    tasks: Iterable[SampleTask],
    *,
    budget: int,
    previously_completed: set[str],
) -> ScheduleResult:
    ordered = sorted(tasks, key=lambda task: (task.scheme, task.sample.sample_id))
    decisions: list[TaskDecision] = []
    approved: list[SampleTask] = []
    pending: list[SampleTask] = []
    reused: list[SampleTask] = []
    deductions: list[dict[str, Any]] = []
    pending_failures: list[Failure] = []
    remaining = int(budget)
    blocked = False

    for task in ordered:
        if blocked:
            pending.append(task)
            pending_failures.append(
                Failure(
                    code=ErrorCode.POINT_BUDGET_PENDING,
                    message="单批点数预算不足，任务保留在待算队列",
                    sample_id=task.sample.sample_id,
                    stage="scheduler",
                    details={"scheme": task.scheme, "sort_key": task.sort_key},
                )
            )
            decisions.append(TaskDecision(task=task, action="pending"))
            continue

        if task.cache_reusable:
            reused.append(task)
            action = (
                "resumed_reuse" if task.sort_key in previously_completed else "cache_reuse"
            )
            decisions.append(TaskDecision(task=task, action=action))
            continue

        if task.points > remaining:
            blocked = True
            pending.append(task)
            pending_failures.append(
                Failure(
                    code=ErrorCode.POINT_BUDGET_PENDING,
                    message="单批点数预算不足，任务保留在待算队列",
                    sample_id=task.sample.sample_id,
                    stage="scheduler",
                    details={
                        "scheme": task.scheme,
                        "sort_key": task.sort_key,
                        "required_points": task.points,
                        "remaining_points": remaining,
                    },
                )
            )
            decisions.append(TaskDecision(task=task, action="pending"))
            continue

        remaining_before = remaining
        remaining -= task.points
        approved.append(task)
        decisions.append(
            TaskDecision(
                task=task,
                action="scheduled",
                points=task.points,
                remaining_before=remaining_before,
                remaining_after=remaining,
            )
        )
        deductions.append(
            {
                "sort_key": task.sort_key,
                "scheme": task.scheme,
                "sample_id": task.sample.sample_id,
                "points": task.points,
                "remaining_before": remaining_before,
                "remaining_after": remaining,
            }
        )

    return ScheduleResult(
        ordered_tasks=tuple(ordered),
        decisions=tuple(decisions),
        approved=tuple(approved),
        pending=tuple(pending),
        reused=tuple(reused),
        budget_deductions=tuple(deductions),
        pending_failures=tuple(pending_failures),
        remaining_budget=remaining,
    )

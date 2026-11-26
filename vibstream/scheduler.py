"""Stable point-budget and per-cluster-quota batch scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .canonical import is_int
from .errors import (
    E_CLUSTER_QUOTA_EXCEEDED,
    E_CONFIG_INVALID,
    E_POINT_BUDGET_PENDING,
    ValidationError,
)


@dataclass(frozen=True)
class ScheduledTask:
    window_id: str
    anchor_arrival_seq: int
    window_index: int
    point_cost: int
    cluster_id: str | None = None
    priority: int = 0
    pending_reason: str | None = None

    def sort_key(self) -> tuple[int, int, int, str]:
        return (self.priority, self.anchor_arrival_seq, self.window_index, self.window_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "anchor_arrival_seq": self.anchor_arrival_seq,
            "window_index": self.window_index,
            "point_cost": self.point_cost,
            "cluster_id": self.cluster_id,
            "priority": self.priority,
            "pending_reason": self.pending_reason,
        }


def _coerce_task(task: ScheduledTask | dict[str, Any]) -> ScheduledTask:
    if isinstance(task, ScheduledTask):
        return task
    if not isinstance(task, dict):
        raise ValidationError(E_CONFIG_INVALID, "task must be an object", "task")
    return ScheduledTask(
        window_id=str(task["window_id"]),
        anchor_arrival_seq=int(task["anchor_arrival_seq"]),
        window_index=int(task["window_index"]),
        point_cost=int(task["point_cost"]),
        cluster_id=task.get("cluster_id"),
        priority=int(task.get("priority", 0)),
        pending_reason=task.get("pending_reason"),
    )


class BatchScheduler:
    """Schedules window work without ever charging an unfinished window twice."""

    def __init__(
        self,
        *,
        batch_point_budget: int,
        cluster_point_quota: int,
        pending: Iterable[ScheduledTask | dict[str, Any]] = (),
        consumed_points_total: int = 0,
        charged_windows: dict[str, int] | None = None,
        batch_number: int = 0,
        errors: Iterable[dict[str, Any]] = (),
    ) -> None:
        if not is_int(batch_point_budget) or batch_point_budget <= 0:
            raise ValidationError(E_CONFIG_INVALID, "batch_point_budget must be positive", "batch_point_budget")
        if not is_int(cluster_point_quota) or cluster_point_quota <= 0:
            raise ValidationError(E_CONFIG_INVALID, "cluster_point_quota must be positive", "cluster_point_quota")
        self.batch_point_budget = batch_point_budget
        self.cluster_point_quota = cluster_point_quota
        self.pending: list[ScheduledTask] = sorted((_coerce_task(task) for task in pending), key=ScheduledTask.sort_key)
        self.consumed_points_total = int(consumed_points_total)
        self.charged_windows: dict[str, int] = dict(charged_windows or {})
        self.batch_number = int(batch_number)
        self.errors: list[dict[str, Any]] = list(errors)
        self.last_processed: list[str] = []
        self.last_pending: list[str] = []

    def enqueue(self, tasks: Iterable[ScheduledTask | dict[str, Any]]) -> None:
        existing = {task.window_id for task in self.pending}
        for raw in tasks:
            task = _coerce_task(raw)
            if task.window_id in existing or task.window_id in self.charged_windows:
                continue
            if task.point_cost <= 0:
                raise ValidationError(E_CONFIG_INVALID, "point_cost must be positive", task.window_id)
            self.pending.append(task)
            existing.add(task.window_id)
        self.pending.sort(key=ScheduledTask.sort_key)

    def run_batch(
        self,
        new_tasks: Iterable[ScheduledTask | dict[str, Any]] = (),
        *,
        budget: int | None = None,
        quota: int | None = None,
        cluster_resolver: Callable[[ScheduledTask], str | None] | None = None,
        execute: Callable[[ScheduledTask], None] | None = None,
    ) -> dict[str, Any]:
        batch_budget = self.batch_point_budget if budget is None else int(budget)
        batch_quota = self.cluster_point_quota if quota is None else int(quota)
        if batch_budget <= 0 or batch_quota <= 0:
            raise ValidationError(E_CONFIG_INVALID, "batch budget and cluster quota must be positive", "scheduler")
        if new_tasks:
            self.enqueue(new_tasks)
        self.batch_number += 1
        batch_errors: list[dict[str, Any]] = []
        processed: list[str] = []
        budget_remaining = batch_budget
        points_by_cluster: dict[str, int] = {}
        blocked_clusters: set[str] = set()
        budget_stopped = False
        still_pending: list[ScheduledTask] = []

        for task in sorted(self.pending, key=ScheduledTask.sort_key):
            if task.window_id in self.charged_windows:
                continue
            if budget_stopped or budget_remaining < task.point_cost:
                budget_stopped = True
                still_pending.append(
                    ScheduledTask(
                        window_id=task.window_id,
                        anchor_arrival_seq=task.anchor_arrival_seq,
                        window_index=task.window_index,
                        point_cost=task.point_cost,
                        cluster_id=task.cluster_id,
                        priority=1,
                        pending_reason=E_POINT_BUDGET_PENDING,
                    )
                )
                continue
            cluster_id = task.cluster_id
            if cluster_resolver is not None:
                cluster_id = cluster_resolver(task)
            if cluster_id is not None and cluster_id in blocked_clusters:
                still_pending.append(
                    ScheduledTask(
                        window_id=task.window_id,
                        anchor_arrival_seq=task.anchor_arrival_seq,
                        window_index=task.window_index,
                        point_cost=task.point_cost,
                        cluster_id=cluster_id,
                        priority=1,
                        pending_reason=E_CLUSTER_QUOTA_EXCEEDED,
                    )
                )
                continue
            if cluster_id is not None and points_by_cluster.get(cluster_id, 0) + task.point_cost > batch_quota:
                blocked_clusters.add(cluster_id)
                batch_errors.append(
                    {
                        "code": E_CLUSTER_QUOTA_EXCEEDED,
                        "batch_number": self.batch_number,
                        "window_id": task.window_id,
                        "window_index": task.window_index,
                        "cluster_id": cluster_id,
                        "cluster_points_before": points_by_cluster.get(cluster_id, 0),
                        "window_point_cost": task.point_cost,
                        "cluster_point_quota": batch_quota,
                        "action": "cluster_paused_for_this_batch",
                    }
                )
                still_pending.append(
                    ScheduledTask(
                        window_id=task.window_id,
                        anchor_arrival_seq=task.anchor_arrival_seq,
                        window_index=task.window_index,
                        point_cost=task.point_cost,
                        cluster_id=cluster_id,
                        priority=1,
                        pending_reason=E_CLUSTER_QUOTA_EXCEEDED,
                    )
                )
                continue
            budget_remaining -= task.point_cost
            self.consumed_points_total += task.point_cost
            self.charged_windows[task.window_id] = task.point_cost
            if cluster_id is not None:
                points_by_cluster[cluster_id] = points_by_cluster.get(cluster_id, 0) + task.point_cost
            processed.append(task.window_id)
            if execute is not None:
                execute(task)

        self.pending = sorted(still_pending, key=ScheduledTask.sort_key)
        self.last_processed = processed
        self.last_pending = [task.window_id for task in self.pending]
        if self.pending:
            budget_pending = [task for task in self.pending if task.pending_reason == E_POINT_BUDGET_PENDING]
            quota_pending = [task for task in self.pending if task.pending_reason == E_CLUSTER_QUOTA_EXCEEDED]
            if budget_pending:
                batch_errors.append(
                    {
                        "code": E_POINT_BUDGET_PENDING,
                        "batch_number": self.batch_number,
                        "window_ids": [task.window_id for task in budget_pending],
                        "point_costs": [task.point_cost for task in budget_pending],
                        "budget_remaining": budget_remaining,
                        "action": "retained_for_next_batch_at_higher_priority",
                    }
                )
            if quota_pending:
                batch_errors.append(
                    {
                        "code": E_CLUSTER_QUOTA_EXCEEDED,
                        "batch_number": self.batch_number,
                        "window_ids": [task.window_id for task in quota_pending],
                        "action": "retained_for_next_batch_with_quota_reset",
                    }
                )
        self.errors.extend(batch_errors)
        return {
            "batch_number": self.batch_number,
            "processed_window_ids": processed,
            "pending_window_ids": [task.window_id for task in self.pending],
            "consumed_points_this_batch": batch_budget - budget_remaining,
            "consumed_points_total": self.consumed_points_total,
            "budget_remaining": budget_remaining,
            "cluster_points_this_batch": dict(sorted(points_by_cluster.items())),
            "errors": batch_errors,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_number": self.batch_number,
            "batch_point_budget": self.batch_point_budget,
            "cluster_point_quota": self.cluster_point_quota,
            "consumed_points_total": self.consumed_points_total,
            "charged_windows": dict(sorted(self.charged_windows.items())),
            "pending": [task.to_dict() for task in sorted(self.pending, key=ScheduledTask.sort_key)],
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchScheduler":
        return cls(
            batch_point_budget=int(data["batch_point_budget"]),
            cluster_point_quota=int(data["cluster_point_quota"]),
            consumed_points_total=int(data["consumed_points_total"]),
            charged_windows={str(key): int(value) for key, value in data.get("charged_windows", {}).items()},
            pending=data.get("pending", []),
            batch_number=int(data.get("batch_number", 0)),
            errors=data.get("errors", []),
        )

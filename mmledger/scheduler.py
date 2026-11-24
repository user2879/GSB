"""Deterministic normalization-budget and per-worker quota scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .errors import (
    E_NORMALIZE_BUDGET_PENDING,
    E_WORKER_QUOTA_EXCEEDED,
    Failure,
    sort_failures,
)
from .models import ScheduleResult, ScheduleTicket


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    worker_id: str
    request_ids: tuple[str, ...]

    @property
    def points(self) -> int:
        return len(self.request_ids)


def schedule_batches(
    batches: Iterable[BatchPlan],
    *,
    budget: int,
    worker_quotas: Mapping[str, int],
    previous_pending: Sequence[str] = (),
) -> ScheduleResult:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    for worker_id, quota in worker_quotas.items():
        if quota < 0:
            raise ValueError(f"quota for {worker_id} must be non-negative")

    batch_map: dict[str, BatchPlan] = {}
    for batch in batches:
        if batch.batch_id in batch_map:
            raise ValueError(f"duplicate batch plan {batch.batch_id}")
        batch_map[batch.batch_id] = batch

    pending_set = {batch_id for batch_id in previous_pending if batch_id in batch_map}
    order = sorted(batch_map, key=lambda batch_id: (0 if batch_id in pending_set else 1, batch_id))
    remaining_budget = int(budget)
    worker_used: dict[str, int] = {worker_id: 0 for worker_id in sorted(worker_quotas)}
    tickets: list[ScheduleTicket] = []
    pending: list[str] = []
    quota_blocked: list[str] = []
    failures: list[Failure] = []
    trace: list[dict[str, Any]] = []
    sequence = 0

    for batch_id in order:
        batch = batch_map[batch_id]
        points = batch.points
        if points <= 0:
            continue
        quota = int(worker_quotas.get(batch.worker_id, 0))
        used = int(worker_used.get(batch.worker_id, 0))
        if used + points > quota:
            quota_blocked.append(batch_id)
            failures.append(
                Failure(
                    E_WORKER_QUOTA_EXCEEDED,
                    f"worker {batch.worker_id} would consume {used + points} points, quota is {quota}",
                    "scheduler",
                    item=batch_id,
                    details={
                        "worker_id": batch.worker_id,
                        "used_points": used,
                        "requested_points": points,
                        "quota_points": quota,
                    },
                )
            )
            trace.append(
                {
                    "event": "quota_blocked",
                    "batch_id": batch_id,
                    "worker_id": batch.worker_id,
                    "used_points": used,
                    "requested_points": points,
                    "quota_points": quota,
                    "budget_remaining": remaining_budget,
                }
            )
            continue
        if remaining_budget < points:
            pending.append(batch_id)
            failures.append(
                Failure(
                    E_NORMALIZE_BUDGET_PENDING,
                    f"normalization budget has {remaining_budget} points, batch needs {points}",
                    "scheduler",
                    item=batch_id,
                    details={
                        "budget_before": remaining_budget,
                        "requested_points": points,
                        "worker_id": batch.worker_id,
                    },
                )
            )
            trace.append(
                {
                    "event": "pending",
                    "batch_id": batch_id,
                    "worker_id": batch.worker_id,
                    "budget_before": remaining_budget,
                    "requested_points": points,
                }
            )
            continue

        ticket = ScheduleTicket(
            sequence=sequence,
            batch_id=batch_id,
            worker_id=batch.worker_id,
            points=points,
            priority_pending=batch_id in pending_set,
            budget_before=remaining_budget,
            budget_after=remaining_budget - points,
            worker_used_before=used,
            worker_used_after=used + points,
        )
        tickets.append(ticket)
        sequence += 1
        for point_index, request_id in enumerate(batch.request_ids):
            remaining_budget -= 1
            used += 1
            trace.append(
                {
                    "event": "deduct",
                    "ticket_sequence": ticket.sequence,
                    "batch_id": batch.batch_id,
                    "worker_id": batch.worker_id,
                    "request_id": request_id,
                    "point_index": point_index,
                    "batch_points_remaining": points - point_index - 1,
                    "budget_remaining": remaining_budget,
                    "worker_points_used": used,
                    "worker_quota_points": quota,
                }
            )
        worker_used[batch.worker_id] = used

    return ScheduleResult(
        tickets=tuple(tickets),
        pending_batches=tuple(sorted(pending)),
        quota_blocked_batches=tuple(sorted(quota_blocked)),
        consumed_budget=int(budget) - remaining_budget,
        trace=tuple(trace),
        failures=tuple(sort_failures(failures)),
    )

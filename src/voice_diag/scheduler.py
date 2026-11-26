from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .errors import E_POINT_BUDGET_PENDING, E_QUOTA_EXCEEDED, QuotaExceededError


def stable_task_key(*, fragment_indices: Iterable[int], window_index: int, stage: str) -> tuple[Any, ...]:
    fragments = tuple(sorted(int(item) for item in fragment_indices))
    stage_order = {"features": 0, "posterior": 1, "attribution": 2}.get(stage, 99)
    return (stage_order, fragments, int(window_index), stage)


@dataclass(frozen=True)
class ScheduledJob:
    cache_id: str
    stage: str
    stable_key: tuple[Any, ...]
    interval_ids: tuple[str, ...]
    points: int
    execute: Callable[[], Any] = field(compare=False, repr=False)
    was_pending: bool = False

    def pending_descriptor(self) -> dict[str, Any]:
        return {
            "cache_id": self.cache_id,
            "stage": self.stage,
            "stable_key": list(self.stable_key),
            "interval_ids": list(self.interval_ids),
            "points": self.points,
            "error_code": E_POINT_BUDGET_PENDING,
        }


@dataclass(frozen=True)
class BatchExecution:
    status: str
    results: dict[str, Any]
    pending: tuple[ScheduledJob, ...]
    consumed_points: int
    workers_used: int
    completion_order: tuple[str, ...]
    interval_usage: dict[str, int] = field(default_factory=dict)
    business_codes: tuple[str, ...] = ()

    def as_summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "completed_cache_ids": sorted(self.results),
            "pending": [job.pending_descriptor() for job in self.pending],
            "consumed_points": self.consumed_points,
            "workers_used": self.workers_used,
            "completion_order": list(self.completion_order),
            "interval_usage": {key: self.interval_usage[key] for key in sorted(self.interval_usage)},
            "business_codes": list(self.business_codes),
        }


def _ordered_jobs(jobs: Iterable[ScheduledJob], pending_ids: set[str]) -> list[ScheduledJob]:
    return sorted(
        jobs,
        key=lambda job: (
            0 if job.was_pending or job.cache_id in pending_ids else 1,
            job.stable_key,
            job.cache_id,
        ),
    )


def run_scheduled_batch(
    jobs: Iterable[ScheduledJob],
    *,
    pending_ids: Iterable[str] = (),
    point_budget: int,
    per_interval_quota: int,
    max_workers: int,
    already_consumed_points: int = 0,
    already_interval_usage: Mapping[str, int] | None = None,
) -> BatchExecution:
    """Run a deterministic, bounded batch of independent window tasks.

    Budget and quota are reserved in the same stable order regardless of worker
    completion timing.  Quota overflow aborts before any task is submitted, so
    no partial batch can leak into a snapshot.  A budget shortfall leaves the
    remaining queue pending for the next batch.
    """

    if point_budget < 0:
        raise ValueError("point_budget must be non-negative")
    if per_interval_quota < 0:
        raise ValueError("per_interval_quota must be non-negative")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    pending_set = {str(item) for item in pending_ids}
    ordered = _ordered_jobs(list(jobs), pending_set)
    selected: list[ScheduledJob] = []
    delayed: list[ScheduledJob] = []
    used = int(already_consumed_points)
    interval_usage: dict[str, int] = {
        str(key): int(value) for key, value in (already_interval_usage or {}).items()
    }
    budget_limit = point_budget
    exhausted = False
    for job in ordered:
        if exhausted:
            delayed.append(job)
            continue
        if job.points < 0:
            raise ValueError("job points must be non-negative")
        next_total = used + job.points
        # A task that cannot fit the current batch budget has not been
        # admitted, so it must not consume or test interval quota.  It stays
        # pending and gets priority on the next batch instead.  This also
        # keeps quota errors independent of where the budget cutoff lands.
        if next_total > budget_limit:
            delayed.append(job)
            exhausted = True
            continue
        next_usage = {
            interval_id: interval_usage.get(interval_id, 0) + job.points
            for interval_id in sorted(set(job.interval_ids))
        }
        overflow = {
            interval_id: amount
            for interval_id, amount in next_usage.items()
            if amount > per_interval_quota
        }
        if overflow:
            raise QuotaExceededError(
                "batch aborted because an interval quota would be exceeded",
                details={
                    "cache_id": job.cache_id,
                    "interval_usage": overflow,
                    "per_interval_quota": per_interval_quota,
                },
            )
        selected.append(job)
        used = next_total
        interval_usage = next_usage
    if not selected and delayed:
        return BatchExecution(
            status=E_POINT_BUDGET_PENDING,
            results={},
            pending=tuple(sorted(delayed, key=lambda job: (job.stable_key, job.cache_id))),
            consumed_points=used,
            workers_used=0,
            completion_order=(),
            interval_usage={key: interval_usage[key] for key in sorted(interval_usage)},
            business_codes=(E_POINT_BUDGET_PENDING,),
        )
    results: dict[str, Any] = {}
    completion_order: list[str] = []
    workers_used = min(max_workers, len(selected))
    if selected:
        with ThreadPoolExecutor(max_workers=workers_used, thread_name_prefix="voice-diag-window") as executor:
            futures = {executor.submit(job.execute): job for job in selected}
            for future in as_completed(futures):
                job = futures[future]
                completion_order.append(job.cache_id)
                results[job.cache_id] = future.result()
    pending = tuple(sorted(delayed, key=lambda job: (job.stable_key, job.cache_id)))
    return BatchExecution(
        status=E_POINT_BUDGET_PENDING if pending else "OK",
        results={key: results[key] for key in sorted(results)},
        pending=pending,
        consumed_points=used,
        workers_used=workers_used,
        completion_order=tuple(completion_order),
        interval_usage={key: interval_usage[key] for key in sorted(interval_usage)},
        business_codes=(E_POINT_BUDGET_PENDING,) if pending else (),
    )


def quota_error_summary(error: QuotaExceededError) -> dict[str, Any]:
    return {
        "code": E_QUOTA_EXCEEDED,
        "message": error.message,
        "details": dict(error.details),
    }

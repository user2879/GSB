from __future__ import annotations

import threading
import time

import pytest

from voice_diag.errors import E_QUOTA_EXCEEDED, QuotaExceededError
from voice_diag.scheduler import ScheduledJob, run_scheduled_batch, stable_task_key


def _job(index: int, *, delay: float = 0.0, intervals: tuple[str, ...] = ("I001",), points: int = 1) -> ScheduledJob:
    def execute() -> int:
        if delay:
            time.sleep(delay)
        return index * 10

    return ScheduledJob(
        cache_id=f"feature:W{index:03d}",
        stage="features",
        stable_key=stable_task_key(fragment_indices=(0,), window_index=index, stage="features"),
        interval_ids=intervals,
        points=points,
        execute=execute,
    )


def test_completion_order_does_not_change_results_or_worker_cap() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def make_job(index: int) -> ScheduledJob:
        def execute() -> int:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01 * (7 - index))
            with lock:
                active -= 1
            return index

        return ScheduledJob(
            cache_id=f"feature:W{index:03d}",
            stage="features",
            stable_key=stable_task_key(fragment_indices=(0,), window_index=index, stage="features"),
            interval_ids=("I001",),
            points=1,
            execute=execute,
        )

    execution = run_scheduled_batch(
        [make_job(index) for index in range(7)],
        point_budget=100,
        per_interval_quota=100,
        max_workers=3,
    )
    assert execution.workers_used == 3
    assert peak <= 3
    assert list(execution.results) == sorted(execution.results)
    assert execution.completion_order != tuple(execution.results) or len(execution.results) < 3
    assert execution.results == {f"feature:W{index:03d}": index for index in range(7)}


def test_budget_pending_is_priority_on_next_batch() -> None:
    jobs = [_job(index) for index in range(5)]
    first = run_scheduled_batch(
        jobs,
        point_budget=2,
        per_interval_quota=100,
        max_workers=2,
    )
    assert sorted(first.results) == ["feature:W000", "feature:W001"]
    pending_ids = {job.cache_id for job in first.pending}
    assert pending_ids == {"feature:W002", "feature:W003", "feature:W004"}
    # Rebuild callables as a resumed process would; persistent queue identity is
    # the cache_id, not the Python closure.
    resumed = [
        ScheduledJob(**{**job.__dict__, "was_pending": job.cache_id in pending_ids})
        for job in jobs
        if job.cache_id in pending_ids
    ]
    second = run_scheduled_batch(
        resumed,
        pending_ids=pending_ids,
        point_budget=2,
        per_interval_quota=100,
        max_workers=2,
    )
    assert set(second.results) == {"feature:W002", "feature:W003"}
    assert {job.cache_id for job in second.pending} == {"feature:W004"}


def test_budget_deferred_job_does_not_trigger_quota_error() -> None:
    executed: list[int] = []

    def job(index: int) -> ScheduledJob:
        return ScheduledJob(
            cache_id=f"feature:W{index:03d}",
            stage="features",
            stable_key=stable_task_key(fragment_indices=(0,), window_index=index, stage="features"),
            interval_ids=("I001",),
            points=1,
            execute=lambda index=index: executed.append(index) or index,
        )

    execution = run_scheduled_batch(
        [job(0), job(1), job(2)],
        point_budget=1,
        per_interval_quota=1,
        max_workers=1,
    )
    assert executed == [0]
    assert execution.status == "E_POINT_BUDGET_PENDING"
    assert [item.cache_id for item in execution.pending] == ["feature:W001", "feature:W002"]


def test_quota_overflow_aborts_before_execution() -> None:
    executed: list[int] = []

    def job(index: int) -> ScheduledJob:
        return ScheduledJob(
            cache_id=f"feature:W{index:03d}",
            stage="features",
            stable_key=stable_task_key(fragment_indices=(0,), window_index=index, stage="features"),
            interval_ids=("I001",),
            points=3,
            execute=lambda index=index: executed.append(index),
        )

    with pytest.raises(QuotaExceededError) as caught:
        run_scheduled_batch(
            [job(0), job(1)],
            point_budget=100,
            per_interval_quota=4,
            max_workers=2,
        )
    assert caught.value.code == E_QUOTA_EXCEEDED
    assert executed == []


def test_stable_key_is_stage_and_fragment_ordered() -> None:
    feature = stable_task_key(fragment_indices=(2, 0), window_index=10, stage="features")
    posterior = stable_task_key(fragment_indices=(2, 0), window_index=10, stage="posterior")
    attribution = stable_task_key(fragment_indices=(2, 0), window_index=10, stage="attribution")
    assert feature < posterior < attribution
    assert feature[1] == (0, 2)

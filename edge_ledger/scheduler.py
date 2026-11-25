from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

from .canonical import stable_key
from .config import PipelineConfig
from .errors import Failure


@dataclass(frozen=True)
class RenderTask:
    sample_id: str
    source_shard: str
    stage: str
    points: int
    sort_key: str
    priority_source: str

    @classmethod
    def make(
        cls,
        sample_id: str,
        source_shard: str,
        points: int,
        *,
        stage: str = "render+score",
        priority_source: str = "current_batch",
    ) -> "RenderTask":
        return cls(
            sample_id=sample_id,
            source_shard=source_shard,
            stage=stage,
            points=points,
            sort_key=stable_key(source_shard, sample_id, stage),
            priority_source=priority_source,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_shard": self.source_shard,
            "stage": self.stage,
            "points": self.points,
            "sort_key": self.sort_key,
            "priority_source": self.priority_source,
        }


@dataclass(frozen=True)
class PendingItem:
    sample_id: str
    source_shard: str
    stage: str
    sort_key: str
    code: str = "E_RENDER_BUDGET_PENDING"
    points: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_shard": self.source_shard,
            "stage": self.stage,
            "sort_key": self.sort_key,
            "code": self.code,
            "points": self.points,
        }


@dataclass(frozen=True)
class ScheduleResult:
    scheduled: tuple[RenderTask, ...]
    pending: tuple[PendingItem, ...]
    quota_failures: tuple[Failure, ...]
    consumed_points: int
    budget_remaining: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheduled": [task.as_dict() for task in self.scheduled],
            "pending": [item.as_dict() for item in self.pending],
            "quota_failures": [failure.as_dict() for failure in self.quota_failures],
            "consumed_points": self.consumed_points,
            "budget_remaining": self.budget_remaining,
        }


def schedule_samples(
    sample_entries: Iterable[tuple[str, str]],
    config: PipelineConfig,
    prior_pending: Iterable[dict[str, Any]] = (),
) -> ScheduleResult:
    """Apply quota and budget constraints in a deterministic stable order.

    `sample_entries` is an iterable of (source_shard, sample_id). Pending samples
    from a previous batch are placed before current-batch work while preserving
    their stable sort key. Render points are charged regardless of whether the
    physical render is later served by cache, so cold and warm runs make the same
    scheduling decisions.
    """

    entry_map = {sample_id: shard for shard, sample_id in sample_entries}
    pending_order: list[str] = []
    for item in prior_pending:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id")
        if isinstance(sample_id, str) and sample_id in entry_map and sample_id not in pending_order:
            pending_order.append(sample_id)
    pending_set = set(pending_order)
    normal_order = [
        sample_id
        for _, sample_id in sorted(
            (shard, sample_id) for sample_id, shard in entry_map.items()
        )
        if sample_id not in pending_set
    ]
    ordered_ids = pending_order + normal_order

    quota_used: dict[str, int] = {}
    quota_failures: list[Failure] = []
    quota_eligible: list[RenderTask] = []
    quota_blocked_shards: set[str] = set()
    for sample_id in ordered_ids:
        shard = entry_map[sample_id]
        task = RenderTask.make(
            sample_id,
            shard,
            config.points_per_render,
            priority_source="prior_pending" if sample_id in pending_set else "current_batch",
        )
        if shard in quota_blocked_shards:
            quota_failures.append(_quota_failure(task))
            continue
        used = quota_used.get(shard, 0)
        if used + task.points > config.source_quota_points:
            quota_blocked_shards.add(shard)
            quota_failures.append(_quota_failure(task))
            continue
        quota_used[shard] = used + task.points
        quota_eligible.append(task)

    consumed = 0
    scheduled: list[RenderTask] = []
    pending: list[PendingItem] = []
    budget_exhausted = False
    for task in quota_eligible:
        if budget_exhausted or consumed + task.points > config.render_point_budget:
            budget_exhausted = True
            pending.append(
                PendingItem(
                    sample_id=task.sample_id,
                    source_shard=task.source_shard,
                    stage=task.stage,
                    sort_key=task.sort_key,
                    points=task.points,
                )
            )
            continue
        consumed += task.points
        scheduled.append(task)

    scheduled.sort(key=lambda task: (task.source_shard, task.sample_id))
    pending.sort(key=lambda item: (item.source_shard, item.sample_id))
    quota_failures.sort(key=Failure.sort_key)
    return ScheduleResult(
        scheduled=tuple(scheduled),
        pending=tuple(pending),
        quota_failures=tuple(quota_failures),
        consumed_points=consumed,
        budget_remaining=max(0, config.render_point_budget - consumed),
    )


def _quota_failure(task: RenderTask) -> Failure:
    return Failure(
        code="E_QUOTA_EXCEEDED",
        stage="schedule",
        message=(
            f"source shard {task.source_shard!r} exceeded its render point quota; "
            "remaining samples in this shard were stopped for this batch"
        ),
        target_kind="sample",
        target_id=task.sample_id,
        details={
            "source_shard": task.source_shard,
            "points": task.points,
            "task_sort_key": task.sort_key,
        },
    )


class ConcurrencyProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


T = TypeVar("T")
R = TypeVar("R")


def run_thread_pool(
    tasks: Iterable[T],
    worker: Callable[[T], R],
    max_workers: int,
    probe: ConcurrencyProbe | None = None,
) -> list[R]:
    task_list = list(tasks)
    if not task_list:
        return []
    worker_count = max(1, min(max_workers, len(task_list)))

    def wrapped(task: T) -> R:
        if probe is not None:
            probe.enter()
        try:
            return worker(task)
        finally:
            if probe is not None:
                probe.leave()

    results: list[tuple[int, R]] = []
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="edge-ledger") as executor:
        future_map = {executor.submit(wrapped, task): index for index, task in enumerate(task_list)}
        for future in as_completed(future_map):
            index = future_map[future]
            results.append((index, future.result()))
    results.sort(key=lambda item: item[0])
    return [result for _, result in results]

"""Deterministic point-budget scheduler backed by a bounded thread pool."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .model import PendingTask, SchedulerEvent


T = TypeVar("T")

STAGE_RANK = {
    "quantization": 1,
    "calibration": 2,
}


def stable_task_key(layer: str, stage: str) -> str:
    """Stable key composed only from the layer name and stage name."""

    return f"{STAGE_RANK.get(stage, 9):02d}:{layer}:{stage}"


@dataclass
class ScheduledTask(Generic[T]):
    layer: str
    stage: str
    points: int
    action: Callable[[], T]
    sort_key: str = field(init=False)
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.points < 0:
            raise ValueError("task points must be non-negative")
        self.sort_key = stable_task_key(self.layer, self.stage)

    def order_key(self) -> tuple[str, int]:
        return (self.sort_key, self.sequence)


@dataclass
class SchedulerRunResult(Generic[T]):
    results: list[tuple[str, T]]
    events: list[SchedulerEvent]
    pending: list[PendingTask]
    batch_count: int
    max_observed_concurrency: int

    def result_map(self) -> dict[str, T]:
        return dict(self.results)


class PointBudgetScheduler:
    """Consume point budgets in stable order, execute work concurrently.

    Budgets are reserved before execution and always in serial task order. The
    worker completion order is irrelevant because outputs are stored by stable
    key and returned in that order. A task that cannot fit in the remaining
    batch budget blocks the head of the queue; jobs are never reordered merely
    to fit smaller tasks.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        point_budget_per_batch: int,
        stage_delay_ms: int = 0,
        batch_callback: Callable[[list[tuple[str, T]], int], None] | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if point_budget_per_batch <= 0:
            raise ValueError("point_budget_per_batch must be positive")
        if stage_delay_ms < 0:
            raise ValueError("stage_delay_ms must be non-negative")
        self.max_workers = max_workers
        self.point_budget_per_batch = point_budget_per_batch
        self.stage_delay_ms = stage_delay_ms
        self.batch_callback = batch_callback

    def run(self, tasks: list[ScheduledTask[T]]) -> SchedulerRunResult[T]:
        queue = sorted(tasks, key=lambda task: task.order_key())
        results: list[tuple[str, T]] = []
        events: list[SchedulerEvent] = []
        pending: list[PendingTask] = []
        batch_number = 0
        maximum_concurrency = 0
        active = 0
        lock = threading.Lock()

        while queue:
            batch_number += 1
            remaining = self.point_budget_per_batch
            selected: list[ScheduledTask[T]] = []
            for position, task in enumerate(queue):
                if task.points > remaining:
                    pending = [
                        PendingTask(
                            sort_key=item.sort_key,
                            layer=item.layer,
                            stage=item.stage,
                            points=item.points,
                            reason=(
                                "task_exceeds_batch_budget"
                                if index == position
                                else "point_budget_exhausted"
                            ),
                        )
                        for index, item in enumerate(queue[position:], start=position)
                    ]
                    break
                selected.append(task)
                remaining -= task.points
                events.append(
                    SchedulerEvent(
                        sort_key=task.sort_key,
                        layer=task.layer,
                        stage=task.stage,
                        points=task.points,
                        batch=batch_number,
                        outcome="deducted",
                    )
                )
            if not selected:
                break

            queue = queue[len(selected) :]
            batch_results: list[tuple[str, T]] = []
            if self.stage_delay_ms:
                time.sleep(self.stage_delay_ms / 1000.0)

            def invoke(task: ScheduledTask[T]) -> T:
                nonlocal active, maximum_concurrency
                with lock:
                    active += 1
                    maximum_concurrency = max(maximum_concurrency, active)
                try:
                    return task.action()
                finally:
                    with lock:
                        active -= 1

            with ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="layerpack"
            ) as executor:
                futures = {
                    executor.submit(invoke, task): task for task in selected
                }
                for future in as_completed(futures):
                    task = futures[future]
                    batch_results.append((task.sort_key, future.result()))
            batch_results.sort(key=lambda item: item[0])
            results.extend(batch_results)
            if self.batch_callback is not None:
                self.batch_callback(list(batch_results), batch_number)

        results.sort(key=lambda item: item[0])
        events.sort(key=lambda event: (event.batch, event.sort_key))
        pending.sort(key=lambda item: item.sort_key)
        return SchedulerRunResult(
            results=results,
            events=events,
            pending=pending,
            batch_count=batch_number if queue or events else 0,
            max_observed_concurrency=maximum_concurrency,
        )

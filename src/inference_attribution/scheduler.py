"""Thread-pool scheduling with global and per-worker concurrency quotas."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from .errors import DiagnosticError


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ScheduleEvent:
    batch_id: str
    worker_id: str
    queue_started_ms: float
    queue_ended_ms: float = 0.0
    run_started_ms: float = 0.0
    run_ended_ms: float = 0.0
    thread_name: str = ""

    @property
    def wait_ms(self) -> float:
        return max(0.0, self.queue_ended_ms - self.queue_started_ms)

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "queue_started_ms": self.queue_started_ms,
            "queue_ended_ms": self.queue_ended_ms,
            "run_started_ms": self.run_started_ms,
            "run_ended_ms": self.run_ended_ms,
            "wait_ms": self.wait_ms,
            "thread_name": self.thread_name,
        }


@dataclass(frozen=True)
class ScheduledResult(Generic[T, R]):
    item: T
    result: R
    event: ScheduleEvent


@dataclass(frozen=True)
class ScheduleReport(Generic[T, R]):
    results: tuple[ScheduledResult[T, R], ...]
    max_global_running: int
    max_worker_running: dict[str, int]

    def events(self) -> tuple[ScheduleEvent, ...]:
        return tuple(result.event for result in self.results)


def _now_ms() -> float:
    return time.monotonic_ns() / 1_000_000.0


class QuotaScheduler:
    def __init__(
        self,
        *,
        global_limit: int,
        worker_quotas: dict[str, int],
    ) -> None:
        if global_limit < 1:
            raise DiagnosticError(
                code="E_SPEC",
                message="global concurrency must be positive",
                object="scheduler",
                field="global_limit",
            )
        for worker_id, quota in worker_quotas.items():
            if quota < 1:
                raise DiagnosticError(
                    code="E_SPEC",
                    message="worker concurrency must be positive",
                    object=worker_id,
                    field="concurrency_quota",
                )
        self.global_limit = global_limit
        self.worker_quotas = dict(worker_quotas)

    def run(
        self,
        items: Iterable[T],
        operation: Callable[[T], R],
        *,
        batch_id: Callable[[T], str],
        worker_id: Callable[[T], str],
    ) -> ScheduleReport[T, R]:
        values = list(items)
        condition = threading.Condition()
        running_global = 0
        running_workers: dict[str, int] = {
            worker: 0 for worker in self.worker_quotas
        }
        max_global = 0
        max_workers: dict[str, int] = {
            worker: 0 for worker in self.worker_quotas
        }
        records: dict[int, ScheduleEvent] = {}

        def invoke(index: int, item: T) -> R:
            event = records[index]
            nonlocal running_global
            item_worker_id = worker_id(item)
            event.run_started_ms = _now_ms()
            event.thread_name = threading.current_thread().name
            try:
                return operation(item)
            finally:
                event.run_ended_ms = _now_ms()
                with condition:
                    running_workers[item_worker_id] -= 1
                    running_global -= 1
                    condition.notify_all()

        executor = ThreadPoolExecutor(
            max_workers=self.global_limit, thread_name_prefix="ia-batch"
        )
        futures: list[object] = [None] * len(values)
        pending = deque(range(len(values)))
        try:
            for index, item in enumerate(values):
                item_worker_id = worker_id(item)
                if item_worker_id not in self.worker_quotas:
                    raise DiagnosticError(
                        code="E_SPEC",
                        message="batch refers to a worker without a concurrency quota",
                        object=batch_id(item),
                        field="worker_id",
                    )
                event = ScheduleEvent(
                    batch_id=batch_id(item),
                    worker_id=item_worker_id,
                    queue_started_ms=_now_ms(),
                )
                records[index] = event
            while pending:
                submitted: list[int] = []
                with condition:
                    for index in list(pending):
                        item_worker_id = worker_id(values[index])
                        if running_global >= self.global_limit:
                            break
                        if running_workers[item_worker_id] >= self.worker_quotas[item_worker_id]:
                            continue
                        pending.remove(index)
                        running_workers[item_worker_id] += 1
                        running_global += 1
                        max_global = max(max_global, running_global)
                        max_workers[item_worker_id] = max(
                            max_workers[item_worker_id],
                            running_workers[item_worker_id],
                        )
                        records[index].queue_ended_ms = _now_ms()
                        submitted.append(index)
                for index in submitted:
                    futures[index] = executor.submit(invoke, index, values[index])
                if not submitted:
                    with condition:
                        if running_global == 0:
                            raise RuntimeError("scheduler has pending work but no ready batch")
                        condition.wait(timeout=0.1)
            results = [future.result() for future in futures]  # type: ignore[union-attr]
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
        scheduled = tuple(
            ScheduledResult(item=values[index], result=results[index], event=records[index])
            for index in range(len(values))
        )
        return ScheduleReport(
            results=scheduled,
            max_global_running=max_global,
            max_worker_running=max_workers,
        )

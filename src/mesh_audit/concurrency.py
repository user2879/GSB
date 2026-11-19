from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

from .errors import BusinessError, ErrorCode


T = TypeVar("T")


class OrderedMemoryBudget:
    """Admission-controlled buffer budget with deterministic ticket ordering."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._condition = threading.Condition()
        self._used = 0
        self._peak = 0
        self._next_ticket = 0
        self._wait_events = 0

    def acquire(self, ticket: int, cost: int) -> None:
        if cost > self.capacity:
            raise BusinessError(
                ErrorCode.E_BUDGET,
                "one evaluation task exceeds the declared memory budget",
                {"task_cost_nodes": cost, "memory_budget_nodes": self.capacity},
            )
        with self._condition:
            waited = False
            while ticket != self._next_ticket or self._used + cost > self.capacity:
                waited = True
                self._condition.wait()
            if waited:
                self._wait_events += 1
            self._next_ticket += 1
            self._used += cost
            self._peak = max(self._peak, self._used)

    def release(self, cost: int) -> None:
        with self._condition:
            self._used -= cost
            self._condition.notify_all()

    def metrics(self) -> dict[str, int]:
        with self._condition:
            return {
                "capacity_nodes": self.capacity,
                "peak_inflight_nodes": self._peak,
                "waited_tasks": self._wait_events,
            }


@dataclass(slots=True)
class EvaluationMetrics:
    concurrency: int
    task_count: int
    peak_inflight_nodes: int
    waited_tasks: int
    budget_capacity_nodes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "concurrency": self.concurrency,
            "task_count": self.task_count,
            "peak_inflight_nodes": self.peak_inflight_nodes,
            "waited_tasks": self.waited_tasks,
            "budget_capacity_nodes": self.budget_capacity_nodes,
        }


class BoundedOrderedEvaluator(Generic[T]):
    """Runs indexed tasks in a bounded pool and merges results in index order."""

    def __init__(self, concurrency: int, memory_budget_nodes: int) -> None:
        self.concurrency = int(concurrency)
        self.budget_capacity = int(memory_budget_nodes)
        self._last_metrics = EvaluationMetrics(
            concurrency=self.concurrency,
            task_count=0,
            peak_inflight_nodes=0,
            waited_tasks=0,
            budget_capacity_nodes=self.budget_capacity,
        )

    @property
    def metrics(self) -> EvaluationMetrics:
        return self._last_metrics

    def map_indexed(
        self,
        tasks: Sequence[tuple[int, Callable[[], T], int]],
    ) -> list[T]:
        if not tasks:
            self._last_metrics = EvaluationMetrics(
                concurrency=self.concurrency,
                task_count=0,
                peak_inflight_nodes=0,
                waited_tasks=0,
                budget_capacity_nodes=self.budget_capacity,
            )
            return []
        ordered = sorted(tasks, key=lambda item: item[0])
        indices = [item[0] for item in ordered]
        if len(set(indices)) != len(indices):
            raise ValueError("evaluation task indices must be unique")
        for index, _, cost in ordered:
            if cost > self.budget_capacity:
                raise BusinessError(
                    ErrorCode.E_BUDGET,
                    "one evaluation task exceeds the declared memory budget",
                    {
                        "task_index": index,
                        "task_cost_nodes": cost,
                        "memory_budget_nodes": self.budget_capacity,
                    },
                )
        budget = OrderedMemoryBudget(self.budget_capacity)
        results: list[object] = [None] * len(ordered)

        def execute(slot: int, ticket: int, function: Callable[[], T], cost: int) -> T:
            budget.acquire(ticket, cost)
            try:
                return function()
            finally:
                budget.release(cost)

        max_workers = max(1, min(self.concurrency, len(ordered)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mesh-eval") as pool:
            futures: list[Future[T]] = []
            for ticket, (_, function, cost) in enumerate(ordered):
                futures.append(pool.submit(execute, ticket, ticket, function, cost))
            for slot, future in enumerate(futures):
                results[slot] = future.result()
        metrics = budget.metrics()
        self._last_metrics = EvaluationMetrics(
            concurrency=max_workers,
            task_count=len(ordered),
            peak_inflight_nodes=metrics["peak_inflight_nodes"],
            waited_tasks=metrics["waited_tasks"],
            budget_capacity_nodes=self.budget_capacity,
        )
        return [result for result in results]  # type: ignore[return-value]

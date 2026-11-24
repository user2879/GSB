"""Bounded global/per-partition finalization scheduler with real wait timing."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .errors import SchedulerError

T = TypeVar("T")


@dataclass(frozen=True)
class ScheduledTask(Generic[T]):
    task_id: str
    partition_key: str
    function: Callable[[], T]
    submitted_ns: int = 0


@dataclass
class TaskOutcome(Generic[T]):
    task_id: str
    value: T | None
    partition_key: str
    wait_ms: float
    attempt_durations_ms: list[float] = field(default_factory=list)
    attempts: int = 0
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "partition_key": self.partition_key,
            "wait_ms": self.wait_ms,
            "attempt_durations_ms": self.attempt_durations_ms,
            "attempts": self.attempts,
            "timed_out": self.timed_out,
            "error": self.error,
        }


@dataclass
class SchedulerStats:
    batches: int = 0
    tasks: int = 0
    max_global_active: int = 0
    max_partition_active: int = 0
    total_wait_ms: float = 0.0
    max_wait_ms: float = 0.0
    total_timeout_attempts: int = 0
    retries: int = 0

    def merge(self, other: "SchedulerStats") -> None:
        self.batches += other.batches
        self.tasks += other.tasks
        self.max_global_active = max(
            self.max_global_active, other.max_global_active
        )
        self.max_partition_active = max(
            self.max_partition_active, other.max_partition_active
        )
        self.total_wait_ms += other.total_wait_ms
        self.max_wait_ms = max(self.max_wait_ms, other.max_wait_ms)
        self.total_timeout_attempts += other.total_timeout_attempts
        self.retries += other.retries

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "tasks": self.tasks,
            "max_global_active": self.max_global_active,
            "max_partition_active": self.max_partition_active,
            "total_wait_ms": round(self.total_wait_ms, 3),
            "max_wait_ms": round(self.max_wait_ms, 3),
            "total_timeout_attempts": self.total_timeout_attempts,
            "retries": self.retries,
        }


class BoundedFinalizationScheduler:
    """A real thread pool with two independent quotas.

    Workers only start a task when both the global slot and the task's partition
    slot are available. Wait start is the monotonic clock at batch submission;
    wait end is the monotonic clock immediately before execution.
    """

    def __init__(
        self,
        *,
        global_limit: int,
        per_partition_limit: int,
        task_timeout_ms: int,
        retries: int,
    ) -> None:
        if global_limit < 1 or per_partition_limit < 1:
            raise SchedulerError(
                "scheduler limits must be positive",
                details={
                    "global_limit": global_limit,
                    "per_partition_limit": per_partition_limit,
                },
            )
        if per_partition_limit > global_limit:
            raise SchedulerError(
                "per-partition limit cannot exceed global limit",
                details={
                    "global_limit": global_limit,
                    "per_partition_limit": per_partition_limit,
                },
            )
        if task_timeout_ms < 0 or retries < 0:
            raise SchedulerError(
                "scheduler timeout and retries must be non-negative",
                details={"task_timeout_ms": task_timeout_ms, "retries": retries},
            )
        self.global_limit = global_limit
        self.per_partition_limit = per_partition_limit
        self.task_timeout_ms = task_timeout_ms
        self.retries = retries

    def run(self, tasks: list[ScheduledTask[T]]) -> tuple[list[TaskOutcome[T]], SchedulerStats]:
        if not tasks:
            return [], SchedulerStats()
        submitted_ns = time.monotonic_ns()
        queue = [
            ScheduledTask(
                task_id=task.task_id,
                partition_key=task.partition_key,
                function=task.function,
                submitted_ns=submitted_ns,
            )
            for task in tasks
        ]
        condition = threading.Condition()
        outcomes: list[TaskOutcome[T] | None] = [None] * len(queue)
        pending = set(range(len(queue)))
        active_partition: dict[str, int] = {}
        active_global = 0
        observed_global = 0
        observed_partition = 0
        timeout_attempts = 0
        retry_count = 0

        def execute_attempt(function: Callable[[], T]) -> tuple[T | None, float, str | None, bool]:
            start = time.monotonic_ns()
            try:
                value = function()
                duration = (time.monotonic_ns() - start) / 1_000_000.0
                timed_out = self.task_timeout_ms > 0 and duration > self.task_timeout_ms
                return value, duration, None, timed_out
            except (TimeoutError, socket.timeout) as exc:
                duration = (time.monotonic_ns() - start) / 1_000_000.0
                return None, duration, str(exc), True
            except Exception as exc:  # noqa: BLE001 - surfaced as structured task failure
                duration = (time.monotonic_ns() - start) / 1_000_000.0
                return None, duration, f"{type(exc).__name__}: {exc}", False

        def worker() -> None:
            nonlocal active_global, observed_global, observed_partition
            nonlocal timeout_attempts, retry_count
            while True:
                with condition:
                    while True:
                        if not pending:
                            return
                        selected: int | None = None
                        for candidate in sorted(pending):
                            partition = queue[candidate].partition_key
                            if active_partition.get(partition, 0) < self.per_partition_limit:
                                selected = candidate
                                break
                        if selected is not None:
                            break
                        condition.wait(timeout=0.01)
                    pending.remove(selected)
                    partition = queue[selected].partition_key
                    active_global += 1
                    active_partition[partition] = active_partition.get(partition, 0) + 1
                    observed_global = max(observed_global, active_global)
                    observed_partition = max(
                        observed_partition, active_partition[partition]
                    )
                    wait_end_ns = time.monotonic_ns()

                task = queue[selected]
                attempt_durations: list[float] = []
                value: T | None = None
                error: str | None = None
                timed_out = False
                attempts = 0
                for attempt in range(1, self.retries + 2):
                    attempts = attempt
                    value, duration, error, attempt_timeout = execute_attempt(task.function)
                    attempt_durations.append(round(duration, 3))
                    if attempt_timeout:
                        timed_out = True
                        timeout_attempts += 1
                    else:
                        break
                    if attempt <= self.retries:
                        retry_count += 1
                outcome = TaskOutcome(
                    task_id=task.task_id,
                    value=value,
                    partition_key=partition,
                    wait_ms=round((wait_end_ns - task.submitted_ns) / 1_000_000.0, 3),
                    attempt_durations_ms=attempt_durations,
                    attempts=attempts,
                    timed_out=timed_out,
                    error=error,
                )
                with condition:
                    outcomes[selected] = outcome
                    active_global -= 1
                    active_partition[partition] -= 1
                    if active_partition[partition] <= 0:
                        active_partition.pop(partition, None)
                    condition.notify_all()

        threads = [
            threading.Thread(target=worker, name=f"finalize-{index}", daemon=True)
            for index in range(min(self.global_limit, len(queue)))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final_outcomes: list[TaskOutcome[T]] = []
        for outcome in outcomes:
            if outcome is None:
                raise SchedulerError(
                    "scheduler completed without an outcome",
                    details={"global_limit": self.global_limit},
                )
            final_outcomes.append(outcome)
        wait_values = [item.wait_ms for item in final_outcomes]
        stats = SchedulerStats(
            batches=1,
            tasks=len(final_outcomes),
            max_global_active=observed_global,
            max_partition_active=observed_partition,
            total_wait_ms=sum(wait_values),
            max_wait_ms=max(wait_values, default=0.0),
            total_timeout_attempts=timeout_attempts,
            retries=retry_count,
        )
        return final_outcomes, stats

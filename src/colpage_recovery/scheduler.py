from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import Budget


def monotonic_ms() -> float:
    return time.monotonic_ns() / 1_000_000.0


@dataclass
class ScheduledTask:
    key: int
    tx_id: str
    label: str
    byte_count: int
    context: Any = None


@dataclass
class TaskTimeline:
    key: int
    tx_id: str
    label: str
    byte_count: int
    wait_started_ms: float
    wait_ended_ms: float | None = None
    worker_started_ms: float | None = None
    worker_ended_ms: float | None = None
    wait_ms: float | None = None
    run_ms: float | None = None
    global_active_at_start: int | None = None
    tx_active_at_start: int | None = None
    thread_name: str | None = None
    status: str = "queued"
    result_digest: str | None = None
    error: str | None = None

    def to_dict(self, origin_ms: float | None = None) -> dict[str, Any]:
        data = {
            "lsn": self.key,
            "tx_id": self.tx_id,
            "page_id": self.label,
            "byte_count": self.byte_count,
            "wait_started_ms": round(self.wait_started_ms, 3),
            "wait_ended_ms": None if self.wait_ended_ms is None else round(self.wait_ended_ms, 3),
            "worker_started_ms": None
            if self.worker_started_ms is None
            else round(self.worker_started_ms, 3),
            "worker_ended_ms": None
            if self.worker_ended_ms is None
            else round(self.worker_ended_ms, 3),
            "wait_ms": None if self.wait_ms is None else round(self.wait_ms, 3),
            "run_ms": None if self.run_ms is None else round(self.run_ms, 3),
            "global_active_at_start": self.global_active_at_start,
            "tx_active_at_start": self.tx_active_at_start,
            "thread_name": self.thread_name,
            "status": self.status,
            "patch_digest": self.result_digest,
            "error": self.error,
        }
        if origin_ms is not None:
            data["wait_start_offset_ms"] = round(self.wait_started_ms - origin_ms, 3)
            if self.worker_started_ms is not None:
                data["worker_start_offset_ms"] = round(self.worker_started_ms - origin_ms, 3)
        return data


@dataclass
class SchedulerResult:
    outcomes: dict[int, Any] = field(default_factory=dict)
    timeline: list[TaskTimeline] = field(default_factory=list)
    slot_peak: int = 0
    origin_ms: float = field(default_factory=monotonic_ms)

    def timeline_dicts(self) -> list[dict[str, Any]]:
        events = [event.to_dict(self.origin_ms) for event in self.timeline]
        return sorted(events, key=lambda item: (item["lsn"], item["tx_id"], item["page_id"]))


def run_quota_schedule(
    tasks: list[ScheduledTask],
    worker: Callable[[ScheduledTask], Any],
    budget: Budget,
) -> SchedulerResult:
    """Run tasks in a real thread pool without bypassing quota accounting.

    Quota counters are owned by the dispatcher thread. A task is submitted only
    when both a global slot and its transaction quota are available.
    """
    global_limit = max(1, budget.global_slots)
    tx_limit = max(1, budget.per_tx_quota)
    result = SchedulerResult()
    enqueued_at = monotonic_ms()
    timeline_by_key: dict[int, TaskTimeline] = {
        task.key: TaskTimeline(
            key=task.key,
            tx_id=task.tx_id,
            label=task.label,
            byte_count=task.byte_count,
            wait_started_ms=enqueued_at,
        )
        for task in tasks
    }
    result.timeline = [timeline_by_key[task.key] for task in tasks]

    active_global = 0
    active_by_tx: dict[str, int] = {}
    peak = 0

    with ThreadPoolExecutor(
        max_workers=global_limit,
        thread_name_prefix="colpage-replay",
    ) as executor:
        pending = list(tasks)
        in_flight: dict[Any, tuple[ScheduledTask, float, int, int]] = {}

        while pending or in_flight:
            next_pending: list[ScheduledTask] = []
            for task in pending:
                tx_active = active_by_tx.get(task.tx_id, 0)
                if active_global >= global_limit or tx_active >= tx_limit:
                    next_pending.append(task)
                    continue

                now = monotonic_ms()
                active_global += 1
                active_by_tx[task.tx_id] = tx_active + 1
                peak = max(peak, active_global)
                event = timeline_by_key[task.key]
                event.wait_ended_ms = now
                event.worker_started_ms = now
                event.wait_ms = now - event.wait_started_ms
                event.global_active_at_start = active_global
                event.tx_active_at_start = active_by_tx[task.tx_id]
                event.status = "running"

                future = executor.submit(worker, task)
                in_flight[future] = (task, now, active_global, active_by_tx[task.tx_id])

            pending = next_pending
            if not in_flight:
                if pending:
                    # The quota checks above always admit the first task when no
                    # work is active, so this is defensive.
                    raise RuntimeError("scheduler made no progress")
                continue

            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                task, _started_at, _global_used, _tx_used = in_flight.pop(future)
                event = timeline_by_key[task.key]
                try:
                    worker_result = future.result()
                except Exception as exc:
                    now = monotonic_ms()
                    event.worker_ended_ms = now
                    event.run_ms = now - (event.worker_started_ms or now)
                    event.status = "failed"
                    event.error = f"{type(exc).__name__}: {exc}"
                    active_global -= 1
                    active_by_tx[task.tx_id] -= 1
                    raise

                now = monotonic_ms()
                event.worker_ended_ms = now
                event.run_ms = now - (event.worker_started_ms or now)
                event.status = "committed"
                event.thread_name = getattr(worker_result, "thread_name", None)
                event.result_digest = getattr(worker_result, "patch_digest", None)
                result.outcomes[task.key] = worker_result
                active_global -= 1
                active_by_tx[task.tx_id] -= 1

    result.slot_peak = peak
    return result

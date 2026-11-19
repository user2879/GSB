"""Deterministic budget-constrained local thread-pool pack scheduler."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import E_WIDTH, fail
from .spec import PackingSpec


class JobLike(Protocol):
    pack_id: str
    source_id: str
    width_used: int


@dataclass(frozen=True)
class ScheduleEntry:
    pack_id: str
    source_id: str
    width_used: int
    queue_start_ms: float
    start_ms: float
    end_ms: float
    wait_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "source_id": self.source_id,
            "width_used": self.width_used,
            "queue_start_ms": self.queue_start_ms,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "wait_ms": self.wait_ms,
        }


@dataclass(frozen=True)
class ScheduleResult:
    entries: tuple[ScheduleEntry, ...]
    max_active_width: int
    global_width_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "sorted_ready_scan",
            "global_width_budget": self.global_width_budget,
            "max_active_width": self.max_active_width,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _now_ms() -> float:
    return round(time.perf_counter_ns() / 1_000_000.0, 3)


def schedule_packs(
    jobs: Iterable[JobLike],
    packing: PackingSpec,
    compute: Callable[[JobLike], Any],
    on_result: Callable[[Any, ScheduleEntry], None] | None = None,
) -> tuple[ScheduleResult, dict[str, Any]]:
    """Run pack jobs under global active-width and per-source active quotas.

    The decision order is the canonical ``pack_id`` order.  Jobs are submitted
    only while the tracked active width and source quota permit it.  Width is
    held from submission until completion, which is conservative and therefore
    safely enforces the hardware/resource budget.
    """

    ordered = sorted(jobs, key=lambda job: job.pack_id)
    for job in ordered:
        if job.width_used > packing.global_width_budget:
            fail(
                E_WIDTH,
                "$.packing.global_width_budget",
                "a pack cannot fit within the global width budget",
                pack_id=job.pack_id,
                source_id=job.source_id,
                pack_width=job.width_used,
                global_width_budget=packing.global_width_budget,
            )
    if not ordered:
        return ScheduleResult((), 0, packing.global_width_budget), {}

    queue_start = _now_ms()
    ready = list(ordered)
    running: dict[Future[Any], tuple[JobLike, float]] = {}
    results: dict[str, Any] = {}
    entries: list[ScheduleEntry] = []
    active_width = 0
    active_by_source: dict[str, int] = {}
    max_active_width = 0
    start_lock = threading.Lock()

    def runner(job: JobLike) -> tuple[Any, float, float]:
        with start_lock:
            start_ms = _now_ms()
        value = compute(job)
        end_ms = _now_ms()
        return value, start_ms, end_ms

    with ThreadPoolExecutor(max_workers=packing.workers, thread_name_prefix="bucketpack") as pool:
        while ready or running:
            progressed = False
            for job in list(ready):
                if len(running) >= packing.workers:
                    break
                if active_width + job.width_used > packing.global_width_budget:
                    continue
                if active_by_source.get(job.source_id, 0) >= packing.pack_quota_per_source:
                    continue
                ready.remove(job)
                future = pool.submit(runner, job)
                running[future] = (job, queue_start)
                active_width += job.width_used
                active_by_source[job.source_id] = active_by_source.get(job.source_id, 0) + 1
                max_active_width = max(max_active_width, active_width)
                progressed = True

            if not running:
                if ready and not progressed:
                    # The pre-check above makes this unreachable for positive
                    # budgets, but preserve a deterministic business error.
                    job = ready[0]
                    fail(
                        E_WIDTH,
                        "$.packing.global_width_budget",
                        "no ready pack can be scheduled",
                        pack_id=job.pack_id,
                        pack_width=job.width_used,
                    )
                continue

            completed, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: running[item][0].pack_id):
                job, queued_at = running.pop(future)
                value, start_ms, end_ms = future.result()
                active_width -= job.width_used
                active_by_source[job.source_id] -= 1
                if active_by_source[job.source_id] == 0:
                    del active_by_source[job.source_id]
                entry = ScheduleEntry(
                    pack_id=job.pack_id,
                    source_id=job.source_id,
                    width_used=job.width_used,
                    queue_start_ms=queued_at,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    wait_ms=round(max(0.0, start_ms - queued_at), 3),
                )
                results[job.pack_id] = value
                entries.append(entry)
                if on_result is not None:
                    on_result(value, entry)

    entries.sort(key=lambda entry: entry.pack_id)
    return (
        ScheduleResult(
            entries=tuple(entries),
            max_active_width=max_active_width,
            global_width_budget=packing.global_width_budget,
        ),
        results,
    )

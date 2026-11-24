"""Deterministic wave scheduler backed by a real ThreadPoolExecutor."""

from __future__ import annotations

import concurrent.futures
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

from .errors import E_FIELD, TunnelError, fail
from .util import monotonic_ms, sha256_json


T = TypeVar("T")


@dataclass(frozen=True)
class ScheduledTask:
    task_uid: str
    message_key: str
    tunnel_id: str
    fragment_index: int
    order: int = 0
    payload: Any = None


@dataclass
class DispatchEvent:
    task_uid: str
    wave: int
    tunnel_id: str
    fragment_index: int
    queued_at_monotonic_ms: float
    dispatched_at_monotonic_ms: float

    @property
    def real_wait_ms(self) -> float:
        return self.dispatched_at_monotonic_ms - self.queued_at_monotonic_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_uid": self.task_uid,
            "wave": self.wave,
            "tunnel_id": self.tunnel_id,
            "fragment_index": self.fragment_index,
            "queued_at_monotonic_ms": self.queued_at_monotonic_ms,
            "dispatched_at_monotonic_ms": self.dispatched_at_monotonic_ms,
            "real_wait_ms": self.real_wait_ms,
        }


@dataclass
class TaskOutcome(Generic[T]):
    task: ScheduledTask
    value: T
    thread_name: str
    worker_pid: int


@dataclass
class SchedulerResult(Generic[T]):
    outcomes: tuple[TaskOutcome[T], ...]
    events: tuple[DispatchEvent, ...]
    deterministic_statistics: dict[str, Any]
    real_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic_statistics": self.deterministic_statistics,
            "real_trace": self.real_trace,
        }


def _stable_task_key(task: ScheduledTask) -> tuple[Any, ...]:
    return (task.message_key, task.order, task.tunnel_id, task.fragment_index, task.task_uid)


def run_scheduler(
    tasks: Iterable[ScheduledTask],
    worker: Callable[[ScheduledTask], T],
    *,
    global_inflight_limit: int,
    per_tunnel_inflight_quota: int,
) -> SchedulerResult[T]:
    if isinstance(global_inflight_limit, bool) or not isinstance(global_inflight_limit, int) or global_inflight_limit <= 0:
        fail(E_FIELD, "scheduler.global_inflight_limit", "global inflight limit must be a positive integer")
    if (
        isinstance(per_tunnel_inflight_quota, bool)
        or not isinstance(per_tunnel_inflight_quota, int)
        or per_tunnel_inflight_quota <= 0
    ):
        fail(E_FIELD, "scheduler.per_tunnel_inflight_quota", "per-tunnel quota must be a positive integer")
    if per_tunnel_inflight_quota > global_inflight_limit:
        fail(
            E_FIELD,
            "scheduler.per_tunnel_inflight_quota",
            "per-tunnel quota must not exceed global inflight limit",
        )

    ordered = sorted(list(tasks), key=_stable_task_key)
    uids = [task.task_uid for task in ordered]
    if len(uids) != len(set(uids)):
        fail(E_FIELD, "scheduler.tasks", "task_uid values must be unique")
    if not ordered:
        statistics = {
            "input_tasks": 0,
            "global_peak_inflight": 0,
            "per_tunnel_peak": {},
            "queued_fragments": 0,
            "total_logical_queue_wait_ms": 0,
            "waves": [],
        }
        signature = sha256_json(statistics)
        statistics["statistics_sha256"] = signature
        return SchedulerResult(
            outcomes=(),
            events=(),
            deterministic_statistics=statistics,
            real_trace={"events": [], "threads_used": [], "wall_clock_semantics": "audit_only"},
        )

    scheduler_started = monotonic_ms()
    queue_started = {task.task_uid: scheduler_started for task in ordered}
    pending = list(ordered)
    active_counts: dict[str, int] = {}
    global_active = 0
    wave_number = 0
    global_peak = 0
    per_tunnel_peak: dict[str, int] = {}
    events: list[DispatchEvent] = []
    outcomes: list[TaskOutcome[T]] = []
    waves: list[dict[str, Any]] = []
    active_futures: dict[concurrent.futures.Future[T], ScheduledTask] = {}
    first_error: BaseException | None = None

    def execute(task: ScheduledTask) -> tuple[T, str, int]:
        value = worker(task)
        return value, threading.current_thread().name, os.getpid()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(global_inflight_limit, len(ordered)),
        thread_name_prefix="tunnel-fwd",
    ) as executor:
        while pending or active_futures:
            admitted_this_wave: list[ScheduledTask] = []
            while len(active_futures) < global_inflight_limit:
                selected_position: int | None = None
                for position, task in enumerate(pending):
                    if active_counts.get(task.tunnel_id, 0) >= per_tunnel_inflight_quota:
                        continue
                    selected_position = position
                    break
                if selected_position is None:
                    break
                task = pending.pop(selected_position)
                future = executor.submit(execute, task)
                active_futures[future] = task
                active_counts[task.tunnel_id] = active_counts.get(task.tunnel_id, 0) + 1
                admitted_this_wave.append(task)
                dispatched_at = monotonic_ms()
                events.append(
                    DispatchEvent(
                        task_uid=task.task_uid,
                        wave=wave_number,
                        tunnel_id=task.tunnel_id,
                        fragment_index=task.fragment_index,
                        queued_at_monotonic_ms=queue_started[task.task_uid],
                        dispatched_at_monotonic_ms=dispatched_at,
                    )
                )

            inflight_after_admission = len(active_futures)
            global_peak = max(global_peak, inflight_after_admission)
            for tunnel_id, count in active_counts.items():
                per_tunnel_peak[tunnel_id] = max(per_tunnel_peak.get(tunnel_id, 0), count)
            waves.append(
                {
                    "wave": wave_number,
                    "admitted_task_uids": [task.task_uid for task in admitted_this_wave],
                    "inflight_peak_after_admission": inflight_after_admission,
                    "queued_after_admission": len(pending),
                }
            )

            if not active_futures:
                if pending:
                    fail(E_FIELD, "scheduler.tasks", "scheduler cannot make progress under the configured limits")
                break

            done, _not_done = concurrent.futures.wait(
                active_futures,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            for future in sorted(done, key=lambda item: _stable_task_key(active_futures[item])):
                task = active_futures.pop(future)
                active_counts[task.tunnel_id] -= 1
                if active_counts[task.tunnel_id] == 0:
                    del active_counts[task.tunnel_id]
                try:
                    value, thread_name, worker_pid = future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    continue
                outcomes.append(
                    TaskOutcome(
                        task=task,
                        value=value,
                        thread_name=thread_name,
                        worker_pid=worker_pid,
                    )
                )
            wave_number += 1

    if first_error is not None:
        if isinstance(first_error, TunnelError):
            raise first_error
        raise first_error

    outcomes.sort(key=lambda item: _stable_task_key(item.task))
    events.sort(key=lambda item: (item.wave, item.task_uid))
    queued_fragments = sum(1 for event in events if event.wave > 0)
    logical_waits = {event.task_uid: float(event.wave) for event in events}
    logical_events = [
        {
            "task_uid": event.task_uid,
            "wave": event.wave,
            "queued_at_logical_monotonic_ms": 0.0,
            "dispatched_at_logical_monotonic_ms": float(event.wave),
            "wait_ms": float(event.wave),
        }
        for event in events
    ]
    deterministic_stats: dict[str, Any] = {
        "input_tasks": len(ordered),
        "global_inflight_limit": global_inflight_limit,
        "per_tunnel_inflight_quota": per_tunnel_inflight_quota,
        "global_peak_inflight": global_peak,
        "per_tunnel_peak": dict(sorted(per_tunnel_peak.items())),
        "queued_fragments": queued_fragments,
        "logical_clock_unit": "millisecond_tick_per_completed_admission_wave",
        "total_logical_queue_wait_ms": sum(logical_waits.values()),
        "queue_wait_by_task": {key: logical_waits[key] for key in sorted(logical_waits)},
        "queue_wait_events": logical_events,
        "waves": waves,
    }
    deterministic_stats["statistics_sha256"] = sha256_json(deterministic_stats)
    real_events = [event.to_dict() for event in events]
    threads_used = sorted({item.thread_name for item in outcomes if item.thread_name})
    return SchedulerResult(
        outcomes=tuple(outcomes),
        events=tuple(events),
        deterministic_statistics=deterministic_stats,
        real_trace={
            "semantics": "real_monotonic_audit_not_part_of_deterministic_signature",
            "scheduler_started_monotonic_ms": scheduler_started,
            "events": real_events,
            "threads_used": threads_used,
            "worker_pids": sorted({item.worker_pid for item in outcomes if item.worker_pid}),
        },
    )

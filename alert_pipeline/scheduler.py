from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .models import ScheduleReport, TimelineEntry
from .util import canonical_json, monotonic_ms


@dataclass(frozen=True)
class ScheduledTask:
    ordinal: int
    task_id: str
    stage: str
    service_id: str
    function: Callable[[], Any]


@dataclass(frozen=True)
class ExecutionResult:
    values: tuple[Any, ...]
    report: ScheduleReport
    error: BaseException | None


class _QuotaState:
    def __init__(self, global_limit: int, per_service_limit: int) -> None:
        self.global_limit = global_limit
        self.per_service_limit = per_service_limit
        self.condition = threading.Condition()
        self.running_global = 0
        self.running_by_service: dict[str, int] = {}
        self.peak_global = 0
        self.peak_by_service: dict[str, int] = {}

    def acquire(self, service_id: str) -> int:
        with self.condition:
            while (
                self.running_global >= self.global_limit
                or self.running_by_service.get(service_id, 0) >= self.per_service_limit
            ):
                self.condition.wait()
            self.running_global += 1
            service_running = self.running_by_service.get(service_id, 0) + 1
            self.running_by_service[service_id] = service_running
            self.peak_global = max(self.peak_global, self.running_global)
            self.peak_by_service[service_id] = max(
                self.peak_by_service.get(service_id, 0), service_running
            )
            return monotonic_ms()

    def release(self, service_id: str) -> None:
        with self.condition:
            self.running_global -= 1
            remaining = self.running_by_service.get(service_id, 0) - 1
            if remaining <= 0:
                self.running_by_service.pop(service_id, None)
            else:
                self.running_by_service[service_id] = remaining
            self.condition.notify_all()


def _validate_limits(global_limit: int, per_service_limit: int) -> None:
    if not isinstance(global_limit, int) or global_limit < 1:
        raise ValueError("global concurrency limit must be at least 1")
    if not isinstance(per_service_limit, int) or per_service_limit < 1:
        raise ValueError("per-service concurrency limit must be at least 1")
    if per_service_limit > global_limit:
        raise ValueError("per-service limit cannot exceed global limit")


def run_tasks(
    tasks: Iterable[ScheduledTask],
    *,
    mode: str,
    global_limit: int,
    per_service_limit: int,
) -> ExecutionResult:
    _validate_limits(global_limit, per_service_limit)
    task_list = sorted(list(tasks), key=lambda task: task.ordinal)
    if len({task.ordinal for task in task_list}) != len(task_list):
        raise ValueError("scheduled task ordinals must be unique")
    if mode == "serial":
        return _run_serial(task_list, global_limit, per_service_limit)
    if mode != "parallel":
        raise ValueError("mode must be 'serial' or 'parallel'")
    return _run_parallel(task_list, global_limit, per_service_limit)


def _run_serial(
    tasks: list[ScheduledTask], global_limit: int, per_service_limit: int
) -> ExecutionResult:
    values: list[Any] = []
    timeline: list[TimelineEntry] = []
    error: BaseException | None = None
    peak_by_service: dict[str, int] = {}
    for task in tasks:
        queued = monotonic_ms()
        started = queued
        try:
            value = task.function()
            outcome = "OK"
            values.append(value)
        except BaseException as exc:
            outcome = getattr(exc, "code", "E_STATE")
            error = error or exc
            values.append(None)
        finished = monotonic_ms()
        peak_by_service[task.service_id] = max(
            peak_by_service.get(task.service_id, 0), 1
        )
        timeline.append(
            TimelineEntry(
                ordinal=task.ordinal,
                task_id=task.task_id,
                stage=task.stage,
                service_id=task.service_id,
                queued_at_ms=queued,
                started_at_ms=started,
                finished_at_ms=finished,
                wait_ms=max(0, started - queued),
                run_ms=max(0, finished - started),
                outcome=outcome,
            )
        )
    report = ScheduleReport(
        mode="serial",
        global_limit=global_limit,
        per_service_limit=per_service_limit,
        timeline=tuple(timeline),
        peak_global=1 if tasks else 0,
        peak_by_service=dict(sorted(peak_by_service.items())),
        business_code=getattr(error, "code", "OK") if error else "OK",
    )
    return ExecutionResult(values=tuple(values), report=report, error=error)


def _run_parallel(
    tasks: list[ScheduledTask], global_limit: int, per_service_limit: int
) -> ExecutionResult:
    quota = _QuotaState(global_limit, per_service_limit)
    values_by_ordinal: dict[int, Any] = {}
    timeline_by_ordinal: dict[int, TimelineEntry] = {}
    errors_by_ordinal: dict[int, BaseException] = {}
    queued_at_by_ordinal = {
        task.ordinal: monotonic_ms()
        for task in tasks
    }

    def execute(task: ScheduledTask) -> None:
        queued = queued_at_by_ordinal[task.ordinal]
        started = quota.acquire(task.service_id)
        try:
            value = task.function()
            values_by_ordinal[task.ordinal] = value
            outcome = "OK"
        except BaseException as exc:
            errors_by_ordinal[task.ordinal] = exc
            outcome = getattr(exc, "code", "E_STATE")
        finally:
            finished = monotonic_ms()
            quota.release(task.service_id)
        timeline_by_ordinal[task.ordinal] = TimelineEntry(
            ordinal=task.ordinal,
            task_id=task.task_id,
            stage=task.stage,
            service_id=task.service_id,
            queued_at_ms=queued,
            started_at_ms=started,
            finished_at_ms=finished,
            wait_ms=max(0, started - queued),
            run_ms=max(0, finished - started),
            outcome=outcome,
        )

    with ThreadPoolExecutor(max_workers=global_limit, thread_name_prefix="alert-pipe") as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            future.result()

    first_ordinal_error = min(errors_by_ordinal) if errors_by_ordinal else None
    error = errors_by_ordinal.get(first_ordinal_error) if first_ordinal_error is not None else None
    values = tuple(values_by_ordinal.get(task.ordinal) for task in tasks)
    timeline = tuple(timeline_by_ordinal[task.ordinal] for task in tasks)
    report = ScheduleReport(
        mode="parallel",
        global_limit=global_limit,
        per_service_limit=per_service_limit,
        timeline=timeline,
        peak_global=quota.peak_global,
        peak_by_service=dict(sorted(quota.peak_by_service.items())),
        business_code=getattr(error, "code", "OK") if error else "OK",
    )
    return ExecutionResult(values=values, report=report, error=error)


def combine_reports(reports: Iterable[ScheduleReport], mode: str, global_limit: int, per_service_limit: int) -> ScheduleReport:
    report_list = list(reports)
    ordinal = 0
    timeline: list[TimelineEntry] = []
    peaks: dict[str, int] = {}
    for report in report_list:
        for entry in report.timeline:
            timeline.append(
                TimelineEntry(
                    ordinal=ordinal,
                    task_id=entry.task_id,
                    stage=entry.stage,
                    service_id=entry.service_id,
                    queued_at_ms=entry.queued_at_ms,
                    started_at_ms=entry.started_at_ms,
                    finished_at_ms=entry.finished_at_ms,
                    wait_ms=entry.wait_ms,
                    run_ms=entry.run_ms,
                    outcome=entry.outcome,
                )
            )
            ordinal += 1
        for service, peak in report.peak_by_service.items():
            peaks[service] = max(peaks.get(service, 0), peak)
    business_code = next(
        (report.business_code for report in report_list if report.business_code != "OK"),
        "OK",
    )
    return ScheduleReport(
        mode=mode,
        global_limit=global_limit,
        per_service_limit=per_service_limit,
        timeline=tuple(timeline),
        peak_global=max((report.peak_global for report in report_list), default=0),
        peak_by_service=dict(sorted(peaks.items())),
        business_code=business_code,
    )


def conclusion_bytes(aggregates: Iterable[Any], ledger: Iterable[Any]) -> bytes:
    payload = {
        "aggregates": [item.to_public_dict() for item in aggregates],
        "ledger": [item.consistency_view() for item in ledger],
    }
    return canonical_json(payload).encode("utf-8")

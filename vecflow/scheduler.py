"""Resource-budgeted, per-task-quota thread-pool scheduler."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import AppError, as_app_error
from .planner import ExecutionPlan, ShardPlan, TaskPlan


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class WorkItem:
    sequence: int
    task_id: str
    shard_id: str
    shard_index: int
    resources: int
    shard: ShardPlan
    task: TaskPlan


@dataclass
class ScheduleRecord:
    sequence: int
    task_id: str
    shard_id: str
    shard_index: int
    start_row: int
    end_row: int
    resources: int
    status: str = "queued"
    queued_at_ms: int = 0
    started_at_ms: int | None = None
    ended_at_ms: int | None = None
    wait_duration_ms: int | None = None
    outcome: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    result: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "start": self.start_row,
            "end": self.end_row,
            "resources": self.resources,
            "status": self.status,
            "queued_at_ms": self.queued_at_ms,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "wait_duration_ms": self.wait_duration_ms,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "result_digest": getattr(self.result, "result_digest", None),
        }


@dataclass(frozen=True)
class ScheduleReport:
    run_id: str
    status: str
    records: tuple[ScheduleRecord, ...]
    resource_events: tuple[dict[str, Any], ...]
    peak_resources: int
    task_peak_concurrency: dict[str, int]
    global_budget: int
    task_quotas: dict[str, int]
    stale: bool = False
    error_code: str | None = None
    error_message: str | None = None

    def records_for_task(self, task_id: str) -> list[ScheduleRecord]:
        return [record for record in self.records if record.task_id == task_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "records": [record.to_dict() for record in self.records],
            "resource_events": list(self.resource_events),
            "peak_resources": self.peak_resources,
            "task_peak_concurrency": dict(sorted(self.task_peak_concurrency.items())),
            "global_budget": self.global_budget,
            "task_quotas": dict(sorted(self.task_quotas.items())),
            "stale": self.stale,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


class RunControl:
    """Shared control surface used to invalidate an in-flight run atomically."""

    def __init__(self, run_id: str, work_items: Iterable[WorkItem]) -> None:
        self.run_id = run_id
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._items = {item.shard_id: item for item in work_items}
        self._statuses = {item.shard_id: "queued" for item in work_items}
        self.reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def status(self, shard_id: str) -> str:
        with self._lock:
            return self._statuses.get(shard_id, "unknown")

    def set_status(self, shard_id: str, status: str) -> None:
        with self._lock:
            current = self._statuses.get(shard_id)
            if current == "stale" and status not in {"stale", "unknown"}:
                return
            if shard_id in self._statuses:
                self._statuses[shard_id] = status

    def invalidate(self, reason: str) -> dict[str, Any]:
        with self._lock:
            self.reason = reason
            self._cancel.set()
            affected_running: list[str] = []
            affected_queued: list[str] = []
            for item in sorted(self._items.values(), key=lambda candidate: candidate.sequence):
                status = self._statuses.get(item.shard_id)
                if status == "running":
                    affected_running.append(item.shard_id)
                elif status == "queued":
                    affected_queued.append(item.shard_id)
                if status in {"running", "queued"}:
                    self._statuses[item.shard_id] = "stale"
            return {
                "run_id": self.run_id,
                "running": sorted(affected_running),
                "queued": sorted(affected_queued),
                "affected": sorted(affected_running + affected_queued),
                "reason": reason,
            }

    def snapshot_statuses(self) -> dict[str, str]:
        with self._lock:
            return dict(sorted(self._statuses.items()))


class SchedulerBudgetError(AppError):
    """E_BUDGET carrying the partial, stable scheduling report for display."""

    def __init__(self, message: str, details: dict[str, Any], report: ScheduleReport) -> None:
        super().__init__("E_BUDGET", message, details)
        self.report = report


def build_work_items(plan: ExecutionPlan) -> tuple[WorkItem, ...]:
    items: list[WorkItem] = []
    sequence = 0
    for task in plan.tasks:
        for shard in task.shards:
            items.append(
                WorkItem(
                    sequence=sequence,
                    task_id=task.task_id,
                    shard_id=shard.shard_id,
                    shard_index=shard.index,
                    resources=shard.resources,
                    shard=shard,
                    task=task,
                )
            )
            sequence += 1
    return tuple(items)


def _completed_batch(futures: set[Future[Any]], future_to_item: dict[Future[Any], WorkItem]) -> list[tuple[Future[Any], WorkItem]]:
    return sorted(
        ((future, future_to_item[future]) for future in futures),
        key=lambda pair: pair[1].sequence,
    )


def run_schedule(
    work_items: Iterable[WorkItem],
    *,
    global_budget: int,
    task_quotas: dict[str, int],
    max_workers: int,
    execute_fn: Callable[[WorkItem], Any],
    control: RunControl | None = None,
    on_start: Callable[[ScheduleRecord], None] | None = None,
    on_batch: Callable[[list[ScheduleRecord]], None] | None = None,
) -> ScheduleReport:
    """Run shards under both resource and per-task concurrency ceilings."""

    items = tuple(work_items)
    if global_budget < 0:
        raise AppError("E_DATA", "global_budget 不能为负数")
    if max_workers < 1:
        raise AppError("E_DATA", "max_workers 必须 >= 1")
    run_id = control.run_id if control is not None else uuid.uuid4().hex
    if control is None:
        control = RunControl(run_id, items)

    queued_at = monotonic_ms()
    records: dict[str, ScheduleRecord] = {
        item.shard_id: ScheduleRecord(
            sequence=item.sequence,
            task_id=item.task_id,
            shard_id=item.shard_id,
            shard_index=item.shard_index,
            start_row=item.shard.start_row,
            end_row=item.shard.end_row,
            resources=item.resources,
            queued_at_ms=queued_at,
        )
        for item in items
    }
    pending: list[WorkItem] = list(items)
    active_futures: dict[Future[Any], WorkItem] = {}
    active_resources = 0
    active_by_task: dict[str, int] = {task_id: 0 for task_id in task_quotas}
    peak_resources = 0
    task_peak = {task_id: 0 for task_id in task_quotas}
    resource_events: list[dict[str, Any]] = []
    stopped = False

    def emit_event(kind: str, item: WorkItem, units: int) -> None:
        resource_events.append(
            {
                "at_ms": monotonic_ms(),
                "kind": kind,
                "active_units": units,
                "shard_id": item.shard_id,
                "task_id": item.task_id,
                "sequence": item.sequence,
            }
        )

    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vecflow")
    try:
        while pending or active_futures:
            if control.cancelled:
                stopped = True
                break

            for item in list(pending):
                quota = task_quotas.get(item.task_id, 0)
                if active_by_task.get(item.task_id, 0) >= quota:
                    continue
                if active_resources + item.resources > global_budget:
                    continue

                record = records[item.shard_id]
                started = monotonic_ms()
                record.started_at_ms = started
                record.wait_duration_ms = max(0, started - record.queued_at_ms)
                record.status = "running"
                control.set_status(item.shard_id, "running")
                if on_start is not None:
                    on_start(record)
                active_resources += item.resources
                active_by_task[item.task_id] = active_by_task.get(item.task_id, 0) + 1
                peak_resources = max(peak_resources, active_resources)
                task_peak[item.task_id] = max(task_peak.get(item.task_id, 0), active_by_task[item.task_id])
                emit_event("start", item, active_resources)
                future = executor.submit(execute_fn, item)
                active_futures[future] = item
                pending.remove(item)

            if not active_futures:
                if pending:
                    for item in pending:
                        record = records[item.shard_id]
                        record.status = "budget_blocked"
                        record.outcome = "blocked"
                        record.error_code = "E_BUDGET"
                        record.error_message = "全局预算或任务并发配额持续不足"
                        control.set_status(item.shard_id, "budget_blocked")
                    stopped = True
                    break
                break

            done, _ = wait(tuple(active_futures), return_when=FIRST_COMPLETED)
            completed_batch: list[ScheduleRecord] = []
            for future, item in _completed_batch(done, active_futures):
                record = records[item.shard_id]
                ended = monotonic_ms()
                record.ended_at_ms = ended
                active_resources -= item.resources
                active_by_task[item.task_id] = max(0, active_by_task.get(item.task_id, 0) - 1)
                emit_event("end", item, active_resources)
                try:
                    result = future.result()
                except BaseException as exc:  # shard failure is isolated
                    error = as_app_error(exc)
                    record.outcome = "error"
                    record.error_code = error.code
                    record.error_message = error.message
                    record.result = error
                else:
                    record.result = result
                    record.outcome = getattr(result, "outcome", "success")
                    record.error_code = getattr(result, "error_code", None)
                    record.error_message = getattr(result, "error_message", None)
                if control.cancelled:
                    record.status = "stale"
                    control.set_status(item.shard_id, "stale")
                else:
                    record.status = "completed" if record.outcome == "success" else "failed"
                    control.set_status(item.shard_id, record.status)
                completed_batch.append(record)
                del active_futures[future]

            if completed_batch and not control.cancelled and on_batch is not None:
                on_batch(completed_batch)

        if stopped and control.cancelled:
            for item in pending:
                records[item.shard_id].status = "stale"
                control.set_status(item.shard_id, "stale")
            # Running shards are allowed to finish so no worker thread leaks. Their
            # results are deliberately discarded and never merged.
            for future, item in list(active_futures.items()):
                try:
                    future.result()
                except BaseException:
                    pass
                record = records[item.shard_id]
                record.ended_at_ms = monotonic_ms()
                record.status = "stale"
                control.set_status(item.shard_id, "stale")
                del active_futures[future]
            pending.clear()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    ordered_records = tuple(sorted(records.values(), key=lambda record: record.sequence))
    if control.cancelled:
        return ScheduleReport(
            run_id=run_id,
            status="stale",
            records=ordered_records,
            resource_events=tuple(sorted(resource_events, key=lambda event: (event["at_ms"], 0 if event["kind"] == "end" else 1, event["sequence"]))),
            peak_resources=peak_resources,
            task_peak_concurrency=task_peak,
            global_budget=global_budget,
            task_quotas=dict(task_quotas),
            stale=True,
            error_code="E_STALE",
            error_message=control.reason or "parameters changed",
        )

    blocked = [record for record in ordered_records if record.status == "budget_blocked"]
    if blocked:
        report = ScheduleReport(
            run_id=run_id,
            status="budget_blocked",
            records=ordered_records,
            resource_events=tuple(sorted(resource_events, key=lambda event: (event["at_ms"], 0 if event["kind"] == "end" else 1, event["sequence"]))),
            peak_resources=peak_resources,
            task_peak_concurrency=task_peak,
            global_budget=global_budget,
            task_quotas=dict(task_quotas),
            error_code="E_BUDGET",
            error_message="全局资源预算或任务并发配额不足，分片持续排队且无法开始",
        )
        raise SchedulerBudgetError(
            "全局资源预算或任务并发配额不足，分片持续排队且无法开始",
            {
                "run_id": run_id,
                "global_budget": global_budget,
                "task_quotas": dict(sorted(task_quotas.items())),
                "blocked_shards": [record.shard_id for record in blocked],
                "records": [record.to_dict() for record in ordered_records],
            },
            report,
        )

    return ScheduleReport(
        run_id=run_id,
        status="completed",
        records=ordered_records,
        resource_events=tuple(sorted(resource_events, key=lambda event: (event["at_ms"], 0 if event["kind"] == "end" else 1, event["sequence"]))),
        peak_resources=peak_resources,
        task_peak_concurrency=task_peak,
        global_budget=global_budget,
        task_quotas=dict(task_quotas),
    )

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import PipelineError, budget_error, digest_error, skew_error
from .execution import (
    ShardOutcome,
    TaskSpec,
    estimate_task_bytes,
    execute_task,
    hot_groups,
)
from .model import Dataset, Query, RunConfig, Table


class MonotonicClock:
    def __init__(self) -> None:
        self.started_ns = time.monotonic_ns()
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        with self._lock:
            return (time.monotonic_ns() - self.started_ns) // 1_000_000


@dataclass
class BudgetEvent:
    seq: int
    event: str
    monotonic_ms: int
    task_id: str
    estimate_bytes: int
    active_tasks: int
    used_bytes: int
    budget_bytes: int
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seq": self.seq,
            "event": self.event,
            "monotonic_ms": self.monotonic_ms,
            "task_id": self.task_id,
            "estimate_bytes": self.estimate_bytes,
            "active_tasks": self.active_tasks,
            "used_bytes": self.used_bytes,
            "budget_bytes": self.budget_bytes,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass
class SkewDecision:
    task_id: str
    action: str
    group_key: list[Any]
    hot_count: int
    threshold: int
    strategy: str
    requested_parts: int
    actual_parts: int
    hot_groups: list[dict[str, Any]]
    children: list[dict[str, Any]]
    safety_limit: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action": self.action,
            "group_key": self.group_key,
            "hot_count": self.hot_count,
            "threshold": self.threshold,
            "strategy": self.strategy,
            "requested_parts": self.requested_parts,
            "actual_parts": self.actual_parts,
            "hot_groups": self.hot_groups,
            "children": self.children,
            "safety_limit": self.safety_limit,
        }


@dataclass
class TaskRecord:
    task: TaskSpec
    outcome: ShardOutcome
    status: str
    included: bool
    children: tuple[str, ...]
    start_ms: int
    end_ms: int
    queue_wait_ms: int
    queue_started_ms: int | None
    queue_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "parent_task_id": self.task.parent_task_id,
            "split_source": self.task.split_source,
            "table": self.task.table_name,
            "table_version": self.task.table_version,
            "table_digest": self.task.table_digest,
            "row_start": self.task.row_start,
            "row_end": self.task.row_end,
            "referenced_columns": list(self.task.referenced_columns),
            "split_eligible": self.task.split_eligible,
            "estimated_bytes": self.task.estimated_bytes,
            "status": self.status,
            "included": self.included,
            "children": list(self.children),
            "rows_scanned": self.outcome.rows_scanned,
            "rows_completed": self.outcome.rows_completed,
            "aggregate": self.outcome.aggregate_state,
            "digest": self.outcome.digest,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "queue_started_ms": self.queue_started_ms,
            "queue_count": self.queue_count,
        }


@dataclass
class ScheduleJournal:
    records: dict[str, TaskRecord] = field(default_factory=dict)
    skew_decisions: list[SkewDecision] = field(default_factory=list)
    budget_events: list[BudgetEvent] = field(default_factory=list)
    peak_memory_bytes: int = 0
    peak_active_tasks: int = 0
    completed_markers: int = 0
    _event_seq: int = 0

    def records_in_order(self) -> list[TaskRecord]:
        return [self.records[task_id] for task_id in sorted(self.records)]

    def included_outcomes(self) -> list[ShardOutcome]:
        return [
            record.outcome
            for record in self.records_in_order()
            if record.included
        ]

    @property
    def split_count(self) -> int:
        return sum(1 for decision in self.skew_decisions if decision.action == "split")

    @property
    def split_children_count(self) -> int:
        return sum(decision.actual_parts for decision in self.skew_decisions if decision.action == "split")


@dataclass(frozen=True)
class ScheduleResult:
    journal: ScheduleJournal
    included_outcomes: tuple[ShardOutcome, ...]


CheckpointCallback = Callable[[int, ScheduleJournal], None]


def _make_children(
    dataset: Dataset,
    parent: TaskSpec,
    requested_parts: int,
) -> tuple[TaskSpec, ...]:
    table = dataset.table(parent.table_name)
    length = parent.row_end - parent.row_start
    actual_parts = min(requested_parts, length)
    if actual_parts < 2:
        return ()
    base_size, remainder = divmod(length, actual_parts)
    children: list[TaskSpec] = []
    cursor = parent.row_start
    for index in range(actual_parts):
        size = base_size + (1 if index < remainder else 0)
        row_end = cursor + size
        child_id = f"{parent.task_id}/s{index + 1:02d}"
        children.append(
            TaskSpec(
                task_id=child_id,
                table_name=parent.table_name,
                table_version=parent.table_version,
                table_digest=parent.table_digest,
                row_start=cursor,
                row_end=row_end,
                referenced_columns=parent.referenced_columns,
                split_eligible=False,
                parent_task_id=parent.task_id,
                split_source=parent.task_id,
                estimated_bytes=estimate_task_bytes(
                    table,
                    parent.referenced_columns,
                    row_end - cursor,
                ),
            )
        )
        cursor = row_end
    if cursor != parent.row_end:
        raise AssertionError("child ranges do not cover the parent range")
    if children[0].row_start != parent.row_start:
        raise AssertionError("first child does not start at parent start")
    for left, right in zip(children, children[1:]):
        if left.row_end != right.row_start:
            raise AssertionError("child ranges are not adjacent")
    return tuple(children)


def _execute_worker(
    dataset: Dataset,
    query: Query,
    task: TaskSpec,
    clock: MonotonicClock,
) -> tuple[ShardOutcome, int, int]:
    started_ms = clock.now_ms()
    outcome = execute_task(dataset, query, task)
    ended_ms = clock.now_ms()
    return outcome, started_ms, ended_ms


def run_scheduler(
    dataset: Dataset,
    query: Query,
    config: RunConfig,
    tasks: Iterable[TaskSpec],
    checkpoint_callback: CheckpointCallback | None = None,
    journal: ScheduleJournal | None = None,
) -> ScheduleResult:
    clock = MonotonicClock()
    if journal is None:
        journal = ScheduleJournal()
    pending = sorted(list(tasks), key=lambda task: task.task_id)
    injection_target = config.inject_digest_task
    if injection_target == "first":
        injection_target = pending[0].task_id if pending else None
    waiting: dict[str, bool] = {task.task_id: False for task in pending}
    wait_started: dict[str, int] = {}
    queue_counts: dict[str, int] = {task.task_id: 0 for task in pending}
    active: dict[Future[tuple[ShardOutcome, int, int]], TaskSpec] = {}
    budget = config.effective_memory_budget()
    used_bytes = 0
    budget_event_seq = 0

    def add_budget_event(
        event: str,
        task: TaskSpec,
        *,
        reason: str | None = None,
    ) -> None:
        nonlocal budget_event_seq
        budget_event_seq += 1
        journal.budget_events.append(
            BudgetEvent(
                seq=budget_event_seq,
                event=event,
                monotonic_ms=clock.now_ms(),
                task_id=task.task_id,
                estimate_bytes=task.estimated_bytes,
                active_tasks=len(active),
                used_bytes=used_bytes,
                budget_bytes=budget,
                reason=reason,
            )
        )

    def maybe_checkpoint() -> None:
        if checkpoint_callback is None:
            return
        if journal.completed_markers % config.ledger_batch_size != 0:
            return
        batch_no = journal.completed_markers // config.ledger_batch_size
        checkpoint_callback(batch_no, journal)

    try:
        with ThreadPoolExecutor(
            max_workers=config.parallelism,
            thread_name_prefix="columnar-shard",
        ) as pool:
            while pending or active:
                launched = False
                remaining: list[TaskSpec] = []
                now_ms = clock.now_ms()
                for task in pending:
                    if len(active) >= config.parallelism:
                        if not waiting[task.task_id]:
                            waiting[task.task_id] = True
                            wait_started[task.task_id] = now_ms
                            queue_counts[task.task_id] += 1
                            add_budget_event("queued", task, reason="parallelism_limit")
                        remaining.append(task)
                        continue
                    if used_bytes + task.estimated_bytes > budget:
                        if not waiting[task.task_id]:
                            waiting[task.task_id] = True
                            wait_started[task.task_id] = now_ms
                            queue_counts[task.task_id] += 1
                            add_budget_event("queued", task, reason="memory_budget")
                        remaining.append(task)
                        continue
                    used_bytes += task.estimated_bytes
                    if used_bytes > budget:
                        raise AssertionError("memory budget invariant violated")
                    journal.peak_memory_bytes = max(journal.peak_memory_bytes, used_bytes)
                    future = pool.submit(_execute_worker, dataset, query, task, clock)
                    active[future] = task
                    add_budget_event("start", task)
                    journal.peak_active_tasks = max(journal.peak_active_tasks, len(active))
                    launched = True
                pending = remaining
                if not active and pending:
                    task = min(pending, key=lambda item: item.estimated_bytes)
                    add_budget_event(
                        "budget_rejected",
                        task,
                        reason="task estimate exceeds available memory budget",
                    )
                    raise budget_error(
                        "task cannot be scheduled because its estimated memory exceeds the budget",
                        task_id=task.task_id,
                        details={
                            "estimated_bytes": task.estimated_bytes,
                            "budget_bytes": budget,
                            "requested_budget_bytes": config.memory_budget_bytes,
                            "injected_budget_shortfall": config.inject_budget_shortfall,
                            "queue_count": queue_counts.get(task.task_id, 0),
                        },
                    )
                if not active:
                    if not launched and pending:
                        raise AssertionError("scheduler made no progress")
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: active[item].task_id):
                    task = active.pop(future)
                    outcome, started_ms, ended_ms = future.result()
                    used_bytes -= task.estimated_bytes
                    if used_bytes < 0:
                        raise AssertionError("memory accounting underflow")
                    add_budget_event("finish", task)
                    queue_wait_ms = 0
                    if waiting.get(task.task_id):
                        queue_wait_ms = max(0, started_ms - wait_started[task.task_id])
                    digest_injection_applies = (
                        injection_target is not None and task.task_id == injection_target
                    )
                    if digest_injection_applies:
                        journal.records[task.task_id] = TaskRecord(
                            task=task,
                            outcome=outcome,
                            status="digest_mismatch",
                            included=False,
                            children=(),
                            start_ms=started_ms,
                            end_ms=ended_ms,
                            queue_wait_ms=queue_wait_ms,
                            queue_started_ms=(
                                wait_started.get(task.task_id)
                                if waiting.get(task.task_id)
                                else None
                            ),
                            queue_count=queue_counts.get(task.task_id, 0),
                        )
                        journal.completed_markers += 1
                        maybe_checkpoint()
                        expected = "injected-corrupt:" + outcome.digest
                        raise digest_error(
                            "injected shard digest mismatch",
                            task_id=task.task_id,
                            details={
                                "expected": expected,
                                "actual": outcome.digest,
                                "injection": config.inject_digest_task,
                            },
                        )
                    detected = hot_groups(outcome, config.split_threshold)
                    children: tuple[TaskSpec, ...] = ()
                    action: str | None = None
                    max_count = detected[0][1] if detected else 0
                    hot_payload = [
                        {"key": list(key), "count": count} for key, count in detected
                    ]
                    safety_rejected = bool(detected and max_count > config.skew_safety_limit)
                    if detected:
                        trigger_key, trigger_count = detected[0]
                        if safety_rejected:
                            action = "rejected_safety"
                        else:
                            children = _make_children(
                                dataset,
                                task,
                                config.split_parts,
                            ) if task.split_eligible else ()
                            action = "split" if children else "observed_after_split"
                        decision = SkewDecision(
                            task_id=task.task_id,
                            action=action,
                            group_key=list(trigger_key),
                            hot_count=trigger_count,
                            threshold=config.split_threshold,
                            strategy=config.split_strategy,
                            requested_parts=config.split_parts,
                            actual_parts=len(children),
                            hot_groups=hot_payload,
                            children=[
                                {
                                    "task_id": child.task_id,
                                    "row_start": child.row_start,
                                    "row_end": child.row_end,
                                }
                                for child in children
                            ],
                            safety_limit=config.skew_safety_limit,
                        )
                        journal.skew_decisions.append(decision)
                    if safety_rejected:
                        record_status = "skew_rejected"
                    elif children:
                        record_status = "split_parent"
                    else:
                        record_status = "completed"
                    included = record_status == "completed"
                    record = TaskRecord(
                        task=task,
                        outcome=outcome,
                        status=record_status,
                        included=included,
                        children=tuple(child.task_id for child in children),
                        start_ms=started_ms,
                        end_ms=ended_ms,
                        queue_wait_ms=queue_wait_ms,
                        queue_started_ms=(
                            wait_started.get(task.task_id)
                            if waiting.get(task.task_id)
                            else None
                        ),
                        queue_count=queue_counts.get(task.task_id, 0),
                    )
                    journal.records[task.task_id] = record
                    journal.completed_markers += 1
                    if children:
                        for child in children:
                            waiting[child.task_id] = False
                            queue_counts[child.task_id] = 0
                            pending.append(child)
                        pending.sort(key=lambda item: item.task_id)
                    maybe_checkpoint()
                    if safety_rejected:
                        raise skew_error(
                            "runtime hot group exceeded the configured skew safety limit",
                            task_id=task.task_id,
                            details={
                                "group_key": list(detected[0][0]),
                                "hot_count": detected[0][1],
                                "safety_limit": config.skew_safety_limit,
                                "threshold": config.split_threshold,
                            },
                        )
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(
            "E_DATA",
            f"scheduler execution failed: {error}",
            location="scheduler",
            details={"exception_type": type(error).__name__},
        ) from error

    included = tuple(
        record.outcome
        for record in journal.records_in_order()
        if record.included
    )
    return ScheduleResult(journal=journal, included_outcomes=included)

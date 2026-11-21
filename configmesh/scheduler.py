"""Budget-constrained parallel dispatch scheduler.

The scheduler owns a global semaphore (concurrency budget) and a per-node
semaphore (per-node quota).  A task may only start after acquiring both, so the
budget can never be bypassed.  Every task records monotonic-clock queue/start/
end milliseconds, queue duration, attempts (retries kept as distinct records
with unchanged single-dispatch semantics) and the running peak concurrency.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import BusinessError, ErrorCode
from .logging_utils import StructuredLogger
from .util import now_ms


@dataclass
class DispatchTask:
    task_id: str
    node_id: str
    version: int
    shard_index: int
    shard_count: int
    payload: dict[str, Any]


@dataclass
class AttemptRecord:
    attempt: int
    started_ms: int
    ended_ms: int
    outcome: str                 # ok | timeout | error
    business_code: str | None
    receipt: dict[str, Any] | None = None
    detail: str | None = None


@dataclass
class TaskRecord:
    task_id: str
    node_id: str
    version: int
    shard_index: int
    shard_count: int
    enqueued_ms: int
    started_ms: int | None = None
    ended_ms: int | None = None
    queue_ms: int | None = None
    peak_concurrency_observed: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    status: str = "queued"       # queued | ok | failed
    receipt: dict[str, Any] | None = None


class Scheduler:
    def __init__(self, global_limit: int, per_node_quota: int,
                 logger: StructuredLogger) -> None:
        if global_limit < 1 or per_node_quota < 1:
            raise BusinessError(ErrorCode.E_GRID,
                                "concurrency limits must be positive integers",
                                [{"global_limit": global_limit,
                                  "per_node_quota": per_node_quota}])
        self.global_limit = global_limit
        self.per_node_quota = per_node_quota
        self.logger = logger
        self._global_gate = threading.Semaphore(global_limit)
        self._node_locks: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()
        self._active = 0
        self.peak_concurrency = 0
        self.records: list[TaskRecord] = []

    def _node_semaphore(self, node_id: str) -> threading.Semaphore:
        with self._lock:
            if node_id not in self._node_locks:
                self._node_locks[node_id] = threading.Semaphore(self.per_node_quota)
            return self._node_locks[node_id]

    def run(self, tasks: list[DispatchTask],
            dispatch_fn: Callable[[NodeDispatchHint], Any]) -> list[TaskRecord]:
        """Run all tasks; dispatch_fn receives (task, attempt) and returns a Receipt."""
        threads: list[threading.Thread] = []
        for task in tasks:
            record = TaskRecord(
                task_id=task.task_id, node_id=task.node_id, version=task.version,
                shard_index=task.shard_index, shard_count=task.shard_count,
                enqueued_ms=now_ms(),
            )
            with self._lock:
                self.records.append(record)
            thread = threading.Thread(
                target=self._run_task,
                args=(task, record, dispatch_fn),
                name=f"dispatch-{task.task_id}",
                daemon=True,
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        self.records.sort(key=lambda r: r.task_id)
        return self.records

    def _run_task(self, task: DispatchTask, record: TaskRecord,
                  dispatch_fn: Callable[..., Any]) -> None:
        node_gate = self._node_semaphore(task.node_id)
        self._global_gate.acquire()
        node_gate.acquire()
        try:
            record.started_ms = now_ms()
            record.queue_ms = record.started_ms - record.enqueued_ms
            with self._lock:
                self._active += 1
                self.peak_concurrency = max(self.peak_concurrency, self._active)
                record.peak_concurrency_observed = self._active
            self.logger.info(
                "task_started", task_id=task.task_id, node_id=task.node_id,
                active=self._active, peak=self.peak_concurrency,
                global_limit=self.global_limit, queue_ms=record.queue_ms,
            )
            self._attempts(task, record, dispatch_fn)
        finally:
            record.ended_ms = now_ms()
            with self._lock:
                self._active -= 1
            node_gate.release()
            self._global_gate.release()
            self.logger.info("task_ended", task_id=task.task_id,
                             node_id=task.node_id, status=record.status,
                             attempts=len(record.attempts),
                             duration_ms=record.ended_ms - (record.started_ms or record.ended_ms))

    def _attempts(self, task: DispatchTask, record: TaskRecord,
                  dispatch_fn: Callable[..., Any]) -> None:
        retry_policy = task.payload.get("_retry_policy", "none")
        max_retries = int(task.payload.get("_max_retries", 0))
        attempt = 0
        while True:
            attempt += 1
            started = now_ms()
            try:
                receipt = dispatch_fn(task, attempt)
            except BusinessError as exc:
                ended = now_ms()
                record.attempts.append(AttemptRecord(
                    attempt=attempt, started_ms=started, ended_ms=ended,
                    outcome="timeout" if exc.code == ErrorCode.E_HANDSHAKE else "error",
                    business_code=exc.code.value, receipt=None, detail=exc.message,
                ))
                self.logger.error(exc.code.value, "dispatch_attempt_failed",
                                  task_id=task.task_id, node_id=task.node_id,
                                  attempt=attempt, detail=exc.message)
                can_retry = (
                    retry_policy == "fixed"
                    and exc.code == ErrorCode.E_HANDSHAKE
                    and attempt <= max_retries
                )
                if can_retry:
                    self.logger.info("dispatch_retry_scheduled", task_id=task.task_id,
                                     node_id=task.node_id, failed_attempt=attempt)
                    continue
                record.status = "failed"
                return
            ended = now_ms()
            record.attempts.append(AttemptRecord(
                attempt=attempt, started_ms=started, ended_ms=ended,
                outcome="ok", business_code=None, receipt=receipt.to_dict(),
            ))
            record.status = "ok"
            record.receipt = receipt.to_dict()
            return


class NodeDispatchHint:
    """Small typed bundle handed to the platform dispatch callback."""

    def __init__(self, task: DispatchTask, attempt: int) -> None:
        self.task = task
        self.attempt = attempt

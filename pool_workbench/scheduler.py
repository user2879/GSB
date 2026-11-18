"""Deterministic parallel scheduler with a virtual clock.

The scheduler drives real pooled SQLite connections from a single
deterministic event loop using virtual (logical) milliseconds derived from the
declared task durations.  SQL statements execute for real, while their
``wb_sleep`` duration is simulated instead of burning wall-clock time.  This
keeps genuine parallel resource semantics (multiple busy connections, quota
contention, queueing, timeouts, rebuilds) yet makes every run bit-for-bit
reproducible, which is what snapshot replay requires.

A background constraint monitor still samples the *real* pool on every
acquisition/release, so any over-limit scheduling would be recorded as a
constraint violation rather than silently allowed.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import READ_WRITE, PoolConfig, Task
from .errors import ErrorCode
from .planner import PLAN_GET_TIMEOUT, PLAN_HOLD_TIMEOUT, Plan, build_plan
from .pool import STATE_BUSY, STATE_IDLE, ConnectionPool, PoolEntry
from .structured_log import StructuredLogger, time_monotonic_ms

OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_GET_TIMEOUT = "GET_TIMEOUT"
OUTCOME_HOLD_TIMEOUT = "HOLD_TIMEOUT"
OUTCOME_EXEC_ERROR = "EXEC_ERROR"
OUTCOME_LABELS = {
    OUTCOME_SUCCESS: "成功",
    OUTCOME_GET_TIMEOUT: "获取超时",
    OUTCOME_HOLD_TIMEOUT: "持有超时",
    OUTCOME_EXEC_ERROR: "执行错误",
}

_INTERRUPT_MARKERS = ("interrupted", "abort")
_CONNECTION_FATAL = (sqlite3.ProgrammingError, sqlite3.DatabaseError)


@dataclass
class Trace:
    task_id: str
    workload: str
    conn_type: str
    slot_seq: int | None
    queue_enter_ms: int
    start_ms: int | None
    end_ms: int | None
    wait_ms: int
    actual_wait_ms: int
    outcome: str
    error_code: str | None
    error_message: str | None = None
    rebuilt_from_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "workload": self.workload,
            "conn_type": self.conn_type, "slot_seq": self.slot_seq,
            "queue_enter_ms": self.queue_enter_ms, "start_ms": self.start_ms,
            "end_ms": self.end_ms, "wait_ms": self.wait_ms,
            "actual_wait_ms": self.actual_wait_ms, "outcome": self.outcome,
            "error_code": self.error_code, "error_message": self.error_message,
            "rebuilt_from_seq": self.rebuilt_from_seq,
        }


@dataclass
class SchedulerResult:
    traces: list[Trace] = field(default_factory=list)
    outcome_counts: dict[str, int] = field(default_factory=dict)
    plan: Plan | None = None
    rebuild_count: int = 0
    constraint_violations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "traces": [trace.to_dict() for trace in self.traces],
            "outcome_counts": dict(self.outcome_counts),
            "rebuild_count": self.rebuild_count,
            "constraint_violations": list(self.constraint_violations),
            "peak_active": self.plan.peak_active if self.plan else 0,
            "peak_workload": self.plan.peak_workload if self.plan else {},
        }


class DeterministicScheduler:
    """Event-loop scheduler that executes SQL on real connections deterministically."""

    def __init__(
        self,
        pool: ConnectionPool,
        logger: StructuredLogger,
        config: PoolConfig,
        failure_injector: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        self._pool = pool
        self._logger = logger
        self._config = config
        self._failure_injector = failure_injector
        self._traces: list[Trace] = []
        self._monitor: list[dict[str, int]] = []
        self._origin = time_monotonic_ms()
        self._exec_lock = threading.Lock()
        self._release_at: dict[int, tuple[int, str, PoolEntry]] = {}

    def _sample_limits(self) -> None:
        active = self._pool.active_count()
        if active > self._config.max_connections:
            self._monitor.append({"kind": "max_connections", "observed": active})
        usage = self._pool.workload_active()
        for workload, used in usage.items():
            if used > self._config.workload_quotas[workload]:
                self._monitor.append(
                    {"kind": "workload_quota", "workload": workload, "observed": used}
                )

    def _open_planned_connections(self, plan: Plan) -> None:
        for event in plan.open_events:
            self._pool.ensure_capacity(event["conn_type"])

    def _execute_on_entry(
        self, task: Task, planned: Any, entry: PoolEntry
    ) -> tuple[str, str | None, str | None, int | None]:
        # wb_sleep becomes a no-op during simulation: duration is accounted for
        # by the virtual clock rather than real sleeping.
        entry.conn.create_function("wb_sleep", 1, lambda milliseconds: int(milliseconds))
        if self._failure_injector is not None:
            self._failure_injector(entry.conn)
        try:
            cursor = entry.conn.execute(task.sql)
            if task.sql.lstrip()[:6].upper() == "SELECT":
                cursor.fetchall()
            if entry.conn_type == READ_WRITE:
                entry.conn.commit()
        except sqlite3.OperationalError as exc:
            if any(marker in str(exc).lower() for marker in _INTERRUPT_MARKERS):
                self._safe_rollback(entry)
                return (OUTCOME_HOLD_TIMEOUT, ErrorCode.E_HOLDTIMEOUT,
                        f"interrupted at statement boundary ({exc})", None)
            return self._handle_failure(task, entry, exc)
        except _CONNECTION_FATAL as exc:
            return self._handle_failure(task, entry, exc)
        except sqlite3.Error as exc:
            self._safe_rollback(entry)
            return OUTCOME_EXEC_ERROR, ErrorCode.E_EXEC, str(exc), None

        if planned.outcome == PLAN_HOLD_TIMEOUT:
            return (OUTCOME_HOLD_TIMEOUT, ErrorCode.E_HOLDTIMEOUT,
                    "declared duration exceeded hold_timeout_ms; "
                    "statement interrupted at its boundary", None)
        return OUTCOME_SUCCESS, None, None, None

    @staticmethod
    def _safe_rollback(entry: PoolEntry) -> None:
        try:
            entry.conn.rollback()
        except sqlite3.Error:
            pass

    def _handle_failure(
        self, task: Task, entry: PoolEntry, exc: BaseException
    ) -> tuple[str, str | None, str | None, int | None]:
        dead_seq = entry.seq
        replacement = self._pool.rebuild_dead(
            entry, reason=f"{type(exc).__name__}: {exc}",
            task_id=task.task_id, replacement_type=entry.conn_type,
        )
        self._safe_rollback(entry)
        self._logger.event(
            "connection_rebuild_observed", task_id=task.task_id,
            dead_seq=dead_seq, replacement_seq=replacement.seq,
        )
        return (OUTCOME_EXEC_ERROR, ErrorCode.E_EXEC,
                f"{type(exc).__name__}: {exc}", dead_seq)

    def run(self, tasks: list[Task], plan: Plan) -> SchedulerResult:
        self._open_planned_connections(plan)
        plan_by_id = plan.by_id()
        virtual_ms = 0

        for task in tasks:
            planned = plan_by_id[task.task_id]
            queue_enter_ms = self._origin + planned.queue_enter_ms
            virtual_ms = max(virtual_ms, planned.queue_enter_ms)

            if planned.outcome == PLAN_GET_TIMEOUT:
                self._traces.append(Trace(
                    task_id=task.task_id, workload=task.workload,
                    conn_type=task.conn_type, slot_seq=None,
                    queue_enter_ms=queue_enter_ms, start_ms=None,
                    end_ms=self._origin + (planned.end_ms or virtual_ms),
                    wait_ms=planned.wait_ms,
                    actual_wait_ms=self._config.acquire_timeout_ms,
                    outcome=OUTCOME_GET_TIMEOUT,
                    error_code=ErrorCode.E_GETTIMEOUT,
                    error_message="connection acquisition exceeded acquire_timeout_ms",
                ))
                self._logger.event("task_finished", task_id=task.task_id,
                                   outcome=OUTCOME_GET_TIMEOUT,
                                   error_code=ErrorCode.E_GETTIMEOUT)
                continue

            # Advance virtual time to the planned start so busy entries whose
            # virtual end has passed are released first.
            self._release_completed(planned.start_ms or virtual_ms)
            entry = self._acquire_deterministic(task, planned)
            if entry is None:
                # Defensive: the planner predicted a slot but the live pool
                # refused, which must be reported as an acquisition timeout.
                self._traces.append(Trace(
                    task_id=task.task_id, workload=task.workload,
                    conn_type=task.conn_type, slot_seq=None,
                    queue_enter_ms=queue_enter_ms, start_ms=None,
                    end_ms=self._origin + virtual_ms,
                    wait_ms=planned.wait_ms,
                    actual_wait_ms=self._config.acquire_timeout_ms,
                    outcome=OUTCOME_GET_TIMEOUT,
                    error_code=ErrorCode.E_GETTIMEOUT,
                    error_message="live pool refused the planned acquisition",
                ))
                continue

            self._sample_limits()
            start_ms = self._origin + (planned.start_ms or virtual_ms)
            with self._exec_lock:
                outcome, code, message, rebuilt_from = self._execute_on_entry(task, planned, entry)
            end_ms = self._origin + (planned.end_ms or virtual_ms)
            self._traces.append(Trace(
                task_id=task.task_id, workload=task.workload,
                conn_type=task.conn_type, slot_seq=entry.seq,
                queue_enter_ms=queue_enter_ms, start_ms=start_ms, end_ms=end_ms,
                wait_ms=planned.wait_ms, actual_wait_ms=planned.wait_ms,
                outcome=outcome, error_code=code, error_message=message,
                rebuilt_from_seq=rebuilt_from,
            ))
            self._logger.event("task_finished", task_id=task.task_id,
                               outcome=outcome, error_code=code, slot_seq=entry.seq)
            # Hold the connection until its planned virtual end, modelling the
            # real busy window so later acquisitions see correct idle state.
            entry.acquired_at_ms = start_ms
            self._release_at[entry.seq] = (planned.end_ms or virtual_ms, task.workload, entry)

        self._release_completed(10 ** 12)
        leftover = [entry.seq for entry in self._pool.entries() if entry.state == STATE_BUSY]
        if leftover:
            self._monitor.append({"kind": "leftover_busy", "slots": leftover})

        counts = {label: 0 for label in OUTCOME_LABELS}
        for trace in self._traces:
            counts[trace.outcome] += 1
        result = SchedulerResult(
            traces=list(self._traces),
            outcome_counts=counts,
            plan=plan,
            rebuild_count=self._pool.rebuild_count,
            constraint_violations=list(self._monitor),
        )
        self._logger.event(
            "scheduler_finished", outcome_counts=counts,
            rebuild_count=result.rebuild_count,
            constraint_violations=result.constraint_violations,
        )
        return result

    def _release_completed(self, now_virtual_ms: int) -> None:
        due = sorted(seq for seq, (end_ms, _w, _e) in self._release_at.items()
                     if end_ms <= now_virtual_ms)
        for seq in due:
            end_ms, workload, entry = self._release_at.pop(seq)
            if entry.state == STATE_BUSY:
                self._pool.release(entry, workload)
                self._sample_limits()

    def _acquire_deterministic(self, task: Task, planned: Any) -> PoolEntry | None:
        try:
            return self._pool.acquire(
                task.task_id, task.workload, task.conn_type,
                preferred_seq=planned.slot_seq,
            )
        except RuntimeError:
            return None


def run_batch(
    config: PoolConfig,
    tasks: list[Task],
    pool: ConnectionPool,
    logger: StructuredLogger,
    failure_injector: Callable[[sqlite3.Connection], None] | None = None,
) -> SchedulerResult:
    plan = build_plan(config, tasks)
    scheduler = DeterministicScheduler(pool, logger, config, failure_injector)
    return scheduler.run(tasks, plan)

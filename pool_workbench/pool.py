"""SQLite connection pool with quota ledger and rebuild accounting.

The pool is a *resource manager*.  Scheduling decisions (which task gets which
connection slot and when) live in :mod:`pool_workbench.scheduler`; the pool
guarantees that the number of live connections never exceeds
``max_connections`` and tracks per-workload usage against declared quotas.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import READ_ONLY, READ_WRITE, PoolConfig
from .structured_log import StructuredLogger, time_monotonic_ms

STATE_IDLE = "IDLE"
STATE_BUSY = "BUSY"
STATE_DEAD = "DEAD"


@dataclass
class PoolEntry:
    seq: int
    conn_type: str
    conn: sqlite3.Connection
    abort_event: threading.Event
    state: str = STATE_IDLE
    acquired_at_ms: int | None = None
    task_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "conn_type": self.conn_type,
            "state": self.state,
            "task_id": self.task_id,
            "acquired_at_ms": self.acquired_at_ms,
        }


@dataclass
class PoolEvent:
    seq: int
    kind: str
    reason: str
    ts_ms: int
    task_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "reason": self.reason,
            "ts_ms": self.ts_ms,
            "task_id": self.task_id,
            "detail": self.detail,
        }


class ConnectionPool:
    def __init__(
        self,
        config: PoolConfig,
        db_path: str,
        connector: Callable[..., sqlite3.Connection],
        logger: StructuredLogger,
    ) -> None:
        self.config = config
        self._db_path = db_path
        self._connector = connector
        self._logger = logger
        self._guard = threading.RLock()
        self._entries: list[PoolEntry] = []
        self._seq_counter = 0
        self._workload_active: dict[str, int] = {
            workload: 0 for workload in config.workload_quotas
        }
        self._events: list[PoolEvent] = []
        self.rebuild_count = 0

    # ----- basic introspection ------------------------------------------------
    @property
    def max_connections(self) -> int:
        return self.config.max_connections

    def entries(self) -> list[PoolEntry]:
        with self._guard:
            return list(self._entries)

    def events(self) -> list[PoolEvent]:
        with self._guard:
            return list(self._events)

    def workload_active(self) -> dict[str, int]:
        with self._guard:
            return dict(self._workload_active)

    def active_count(self) -> int:
        with self._guard:
            return sum(1 for entry in self._entries if entry.state == STATE_BUSY)

    def live_count(self) -> int:
        with self._guard:
            return len(self._entries)

    def _record(self, kind: str, reason: str, seq: int = -1,
                task_id: str | None = None, **detail: Any) -> PoolEvent:
        event = PoolEvent(
            seq=seq,
            kind=kind,
            reason=reason,
            ts_ms=time_monotonic_ms(),
            task_id=task_id,
            detail=detail,
        )
        self._events.append(event)
        self._logger.event(
            "pool_event",
            kind=kind,
            reason=reason,
            seq=seq,
            task_id=task_id,
            **detail,
        )
        return event

    # ----- connection lifecycle ----------------------------------------------
    def _new_entry(self, conn_type: str) -> PoolEntry:
        # Caller must hold _guard and have headroom.
        self._seq_counter += 1
        seq = self._seq_counter
        read_only = conn_type == READ_ONLY
        abort_event = threading.Event()
        conn = self._connector(self._db_path, read_only, abort_event)
        entry = PoolEntry(seq=seq, conn_type=conn_type, conn=conn,
                          abort_event=abort_event)
        self._entries.append(entry)
        self._record("connection_opened", f"opened {conn_type} connection", seq=seq)
        return entry

    def ensure_capacity(self, conn_type: str) -> PoolEntry:
        """Open a new connection when the pool has headroom."""
        with self._guard:
            if len(self._entries) >= self.config.max_connections:
                raise RuntimeError("connection pool is already at capacity")
            return self._new_entry(conn_type)

    def acquire(self, task_id: str, workload: str, conn_type: str,
                preferred_seq: int | None = None) -> PoolEntry:
        """Reserve a compatible idle entry and book the workload quota."""
        with self._guard:
            idle = [entry for entry in self._entries if entry.state == STATE_IDLE]
            entry: PoolEntry | None = None
            if preferred_seq is not None:
                entry = next((item for item in idle if item.seq == preferred_seq
                              and self._compatible(item.conn_type, conn_type)), None)
            if entry is None:
                same_type = [item for item in idle
                             if self._compatible(item.conn_type, conn_type)]
                # Deterministic lowest-seq assignment, matching the planner.
                same_type.sort(key=lambda item: item.seq)
                entry = same_type[0] if same_type else None
            if entry is None:
                raise RuntimeError("no compatible idle connection is available")
            if self._workload_active.get(workload, 0) >= self.config.workload_quotas[workload]:
                raise RuntimeError(f"workload {workload} quota exhausted")
            if not entry.lock.acquire(blocking=False):  # defensive, entry is idle
                raise RuntimeError("connection lock unexpectedly held")
            entry.state = STATE_BUSY
            entry.task_id = task_id
            entry.acquired_at_ms = time_monotonic_ms()
            entry.abort_event.clear()
            self._workload_active[workload] += 1
            self._record(
                "connection_acquired",
                f"task {task_id} acquired {entry.conn_type} connection",
                seq=entry.seq,
                task_id=task_id,
                workload=workload,
                workload_active=self._workload_active[workload],
            )
            return entry

    @staticmethod
    def _compatible(entry_type: str, requested: str) -> bool:
        if requested == READ_ONLY:
            return True
        return entry_type == READ_WRITE

    def release(self, entry: PoolEntry, workload: str) -> None:
        with self._guard:
            if entry.state != STATE_BUSY:
                return
            entry.state = STATE_IDLE
            entry.task_id = None
            entry.acquired_at_ms = None
            self._workload_active[workload] = max(
                0, self._workload_active.get(workload, 0) - 1
            )
            self._record(
                "connection_released",
                f"released connection after workload {workload}",
                seq=entry.seq,
                workload=workload,
                workload_active=self._workload_active[workload],
            )
            try:
                entry.lock.release()
            except RuntimeError:
                pass

    def rebuild_dead(self, entry: PoolEntry, reason: str, task_id: str,
                     replacement_type: str) -> PoolEntry:
        """Remove a failed connection, then open its replacement.

        The dead entry is removed *before* opening the replacement, so the pool
        can never transiently hold more than ``max_connections`` connections.
        """
        with self._guard:
            self._record("connection_failed", reason, seq=entry.seq, task_id=task_id)
            try:
                entry.conn.close()
            except sqlite3.Error:
                pass
            if entry in self._entries:
                self._entries.remove(entry)
            entry.state = STATE_DEAD
            try:
                entry.lock.release()
            except RuntimeError:
                pass
            replacement = self._new_entry(replacement_type)
            self.rebuild_count += 1
            self._record(
                "connection_rebuilt",
                f"rebuilt dead connection after: {reason}",
                seq=replacement.seq,
                task_id=task_id,
                rebuild_count=self.rebuild_count,
                replaced_seq=entry.seq,
            )
            return replacement

    # ----- snapshots ----------------------------------------------------------
    def pool_snapshot(self) -> dict[str, Any]:
        with self._guard:
            return {
                "config": {
                    "max_connections": self.config.max_connections,
                    "workload_quotas": dict(self.config.workload_quotas),
                    "hold_timeout_ms": self.config.hold_timeout_ms,
                    "acquire_timeout_ms": self.config.acquire_timeout_ms,
                    "param_version": self.config.param_version,
                },
                "seq_counter": self._seq_counter,
                "rebuild_count": self.rebuild_count,
                "workload_active": dict(self._workload_active),
                "connections": [entry.snapshot() for entry in self._entries],
                "events": [event.to_dict() for event in self._events],
            }

    def close(self) -> None:
        with self._guard:
            for entry in self._entries:
                try:
                    entry.conn.close()
                except sqlite3.Error:
                    pass
            self._entries.clear()
            self._record("pool_closed", "all connections closed")

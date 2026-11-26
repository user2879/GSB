from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class MonotonicClock:
    def now_ms(self) -> int:
        raise NotImplementedError


class SystemMonotonicClock(MonotonicClock):
    """Real OS monotonic clock, normalized to non-negative elapsed milliseconds."""

    def __init__(self) -> None:
        self._start_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        return max(0, (time.monotonic_ns() - self._start_ns) // 1_000_000)


class ScriptedMonotonicClock(MonotonicClock):
    """Thread-safe monotonic millisecond source used for byte-reproducible ledgers."""

    def __init__(self, start_ms: int = 0) -> None:
        self._value = start_ms
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        with self._lock:
            return self._value

    def advance(self, milliseconds: int) -> int:
        if milliseconds < 0:
            raise ValueError("monotonic clock cannot move backwards")
        with self._lock:
            self._value += milliseconds
            return self._value

    def tick(self) -> int:
        return self.advance(1)


@dataclass
class QueueTicket:
    batch_id: str
    connection_id: str
    sequence: int
    wait_start_ms: int
    admitted: bool = False
    queued: bool = False
    wait_end_ms: int | None = None
    wait_ms: int | None = None

    def admit(self, end_ms: int) -> None:
        self.admitted = True
        self.wait_end_ms = end_ms
        self.wait_ms = max(0, end_ms - self.wait_start_ms)


@dataclass
class SchedulerEvent:
    sequence: int
    at_ms: int
    kind: str
    batch_id: str
    connection_id: str
    global_in_flight: int
    connection_in_flight: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "at_ms": self.at_ms,
            "kind": self.kind,
            "batch_id": self.batch_id,
            "connection_id": self.connection_id,
            "global_in_flight": self.global_in_flight,
            "connection_in_flight": self.connection_in_flight,
            "details": self.details,
        }


class InFlightScheduler:
    def __init__(
        self,
        global_limit: int,
        connection_quota: int | dict[str, int],
        clock: MonotonicClock | None = None,
    ) -> None:
        if global_limit < 1:
            raise ValueError("global_limit must be positive")
        self.global_limit = global_limit
        self.default_quota = connection_quota if isinstance(connection_quota, int) else None
        self.quotas = {} if isinstance(connection_quota, int) else dict(connection_quota)
        self.clock = clock or ScriptedMonotonicClock()
        self._condition = threading.Condition()
        self._global_in_flight = 0
        self._conn_in_flight: dict[str, int] = {}
        self._waiting: deque[QueueTicket] = deque()
        self._sequence = 0
        self.events: list[SchedulerEvent] = []

    def quota_for(self, connection_id: str) -> int:
        return self.quotas.get(connection_id, self.default_quota or self.global_limit)

    def _can_admit(self, connection_id: str) -> bool:
        return (
            self._global_in_flight < self.global_limit
            and self._conn_in_flight.get(connection_id, 0) < self.quota_for(connection_id)
        )

    def _record(self, kind: str, ticket: QueueTicket, **details: Any) -> None:
        self.events.append(
            SchedulerEvent(
                sequence=ticket.sequence,
                at_ms=self.clock.now_ms(),
                kind=kind,
                batch_id=ticket.batch_id,
                connection_id=ticket.connection_id,
                global_in_flight=self._global_in_flight,
                connection_in_flight=self._conn_in_flight.get(ticket.connection_id, 0),
                details=details,
            )
        )

    def try_acquire(self, batch_id: str, connection_id: str) -> QueueTicket:
        with self._condition:
            self._sequence += 1
            now = self.clock.now_ms()
            ticket = QueueTicket(batch_id, connection_id, self._sequence, now)
            if self._can_admit(connection_id):
                self._admit_locked(ticket, now, queued=False)
            else:
                ticket.queued = True
                self._waiting.append(ticket)
                self._record("wait_started", ticket, queue_length=len(self._waiting))
            return ticket

    def _admit_locked(self, ticket: QueueTicket, now: int, queued: bool) -> None:
        self._global_in_flight += 1
        self._conn_in_flight[ticket.connection_id] = self._conn_in_flight.get(ticket.connection_id, 0) + 1
        ticket.admit(now)
        ticket.queued = queued or ticket.wait_start_ms != now
        self._record("admitted", ticket, queued=ticket.queued, wait_ms=ticket.wait_ms)

    def acquire(self, batch_id: str, connection_id: str, block: bool = True) -> QueueTicket | None:
        ticket = self.try_acquire(batch_id, connection_id)
        if ticket.admitted:
            return ticket
        if not block:
            return ticket
        with self._condition:
            while not ticket.admitted:
                self._condition.wait(timeout=0.01)
        return ticket

    def release(self, ticket: QueueTicket) -> list[QueueTicket]:
        admitted: list[QueueTicket] = []
        with self._condition:
            if not ticket.admitted:
                raise RuntimeError("cannot release a ticket that was never admitted")
            self._global_in_flight = max(0, self._global_in_flight - 1)
            self._conn_in_flight[ticket.connection_id] = max(0, self._conn_in_flight.get(ticket.connection_id, 0) - 1)
            self._record("released", ticket)
            remaining: deque[QueueTicket] = deque()
            while self._waiting:
                candidate = self._waiting.popleft()
                if candidate.admitted:
                    continue
                if self._can_admit(candidate.connection_id):
                    self._admit_locked(candidate, self.clock.now_ms(), queued=True)
                    admitted.append(candidate)
                else:
                    remaining.append(candidate)
            while remaining:
                self._waiting.appendleft(remaining.pop())
            self._condition.notify_all()
        return admitted

    def trajectory(self) -> list[dict[str, Any]]:
        with self._condition:
            return [event.to_json() for event in sorted(self.events, key=lambda e: (e.sequence, e.kind))]

    def peak_in_flight(self) -> dict[str, int]:
        peak_global = 0
        peak_by_conn: dict[str, int] = {}
        for event in self.events:
            peak_global = max(peak_global, event.global_in_flight)
            peak_by_conn[event.connection_id] = max(
                peak_by_conn.get(event.connection_id, 0), event.connection_in_flight
            )
        return {"global": peak_global, **{f"connection:{k}": v for k, v in sorted(peak_by_conn.items())}}


def run_callable_with_ticket(
    scheduler: InFlightScheduler,
    batch_id: str,
    connection_id: str,
    callback: Callable[[QueueTicket], T],
) -> T:
    ticket = scheduler.acquire(batch_id, connection_id, block=True)
    assert ticket is not None and ticket.admitted
    try:
        return callback(ticket)
    finally:
        scheduler.release(ticket)


def deterministic_queue_demo() -> dict[str, Any]:
    clock = ScriptedMonotonicClock(0)
    scheduler = InFlightScheduler(global_limit=2, connection_quota={"conn-a": 1, "conn-b": 1}, clock=clock)
    a1 = scheduler.try_acquire("batch-a1", "conn-a")
    a2 = scheduler.try_acquire("batch-a2", "conn-a")
    b1 = scheduler.try_acquire("batch-b1", "conn-b")
    b2 = scheduler.try_acquire("batch-b2", "conn-b")
    clock.advance(3)
    scheduler.release(a1)
    clock.advance(2)
    scheduler.release(b1)
    assert a2.admitted and a2.wait_ms == 3
    assert b2.admitted and b2.wait_ms == 5
    scheduler.release(a2)
    scheduler.release(b2)
    return {
        "global_limit": 2,
        "connection_quotas": {"conn-a": 1, "conn-b": 1},
        "peak_in_flight": scheduler.peak_in_flight(),
        "tickets": [
            {
                "batch_id": ticket.batch_id,
                "connection_id": ticket.connection_id,
                "queued": ticket.queued,
                "wait_start_ms": ticket.wait_start_ms,
                "wait_end_ms": ticket.wait_end_ms,
                "wait_ms": ticket.wait_ms,
            }
            for ticket in [a1, a2, b1, b2]
        ],
        "trajectory": scheduler.trajectory(),
    }

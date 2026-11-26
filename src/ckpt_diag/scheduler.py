"""Requirement 6: parallel batch scheduler under three real constraints.

Constraints enforced before a batch starts:

1. global resource-unit budget (sum of in-flight ``cost_units``);
2. per-shard active lease count <= lease_cap;
3. per-shard active batch count <= concurrency_quota.

Batches that cannot be admitted block on a :class:`threading.Condition` (a real
quota gate, never an artificial ``sleep``) and record monotonic-clock
wait start/end in milliseconds. Work runs in a real thread pool.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from threading import Condition
from typing import Callable

from .clock import MonotonicClock
from .lease import Clock


@dataclass
class BatchTicket:
    batch_id: str
    shard_id: str
    cost_units: int
    wait_start_ms: int = 0
    wait_end_ms: int = 0
    start_ms: int = 0
    finish_ms: int = 0
    status: str = "queued"
    result: object = None
    error: str = ""

    @property
    def wait_ms(self) -> int:
        return self.wait_end_ms - self.wait_start_ms


@dataclass
class _ShardCounters:
    active_leases: int = 0
    active_batches: int = 0
    lease_cap: int = 0
    concurrency_quota: int = 0


class Scheduler:
    def __init__(self, *, global_budget: int,
                 shard_limits: dict[str, tuple[int, int]],
                 max_workers: int,
                 clock: Clock | None = None) -> None:
        self.clock = clock or MonotonicClock().now_ms
        self.global_budget = global_budget
        self.in_flight_units = 0
        self.counters = {
            shard_id: _ShardCounters(lease_cap=lease_cap,
                                     concurrency_quota=quota)
            for shard_id, (lease_cap, quota) in shard_limits.items()
        }
        self._cond = Condition()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="ckpt-batch")
        self.tickets: dict[str, BatchTicket] = {}

    def _admittable(self, ticket: BatchTicket) -> bool:
        counters = self.counters[ticket.shard_id]
        return (
            self.in_flight_units + ticket.cost_units <= self.global_budget
            and counters.active_leases + ticket.cost_units <= counters.lease_cap
            and counters.active_batches + 1 <= counters.concurrency_quota
        )

    def _acquire(self, ticket: BatchTicket) -> None:
        ticket.wait_start_ms = self.clock()
        with self._cond:
            while not self._admittable(ticket):
                # Real blocking on the quota gate; no artificial delay.
                self._cond.wait()
            ticket.wait_end_ms = self.clock()
            counters = self.counters[ticket.shard_id]
            self.in_flight_units += ticket.cost_units
            counters.active_leases += ticket.cost_units
            counters.active_batches += 1
            ticket.start_ms = self.clock()
            ticket.status = "running"

    def _release(self, ticket: BatchTicket) -> None:
        with self._cond:
            counters = self.counters[ticket.shard_id]
            self.in_flight_units -= ticket.cost_units
            counters.active_leases -= ticket.cost_units
            counters.active_batches -= 1
            ticket.finish_ms = self.clock()
            self._cond.notify_all()

    def submit(self, ticket: BatchTicket, job: Callable[[BatchTicket], object]) -> Future:
        if ticket.shard_id not in self.counters:
            raise KeyError(f"unknown shard {ticket.shard_id}")
        self.tickets[ticket.batch_id] = ticket

        def runner() -> object:
            self._acquire(ticket)
            try:
                ticket.result = job(ticket)
                ticket.status = "completed"
                return ticket.result
            except Exception as exc:  # propagated via the future
                ticket.status = "failed"
                ticket.error = str(exc)
                raise
            finally:
                self._release(ticket)

        return self._pool.submit(runner)

    def update_quota(self, shard_id: str, lease_cap: int, concurrency_quota: int) -> None:
        with self._cond:
            counters = self.counters[shard_id]
            counters.lease_cap = lease_cap
            counters.concurrency_quota = concurrency_quota
            self._cond.notify_all()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def wait_report(self) -> list[dict]:
        return [
            {
                "batch_id": ticket.batch_id,
                "shard_id": ticket.shard_id,
                "cost_units": ticket.cost_units,
                "wait_start_ms": ticket.wait_start_ms,
                "wait_end_ms": ticket.wait_end_ms,
                "wait_ms": ticket.wait_ms,
                "status": ticket.status,
            }
            for ticket in sorted(self.tickets.values(), key=lambda t: t.batch_id)
        ]

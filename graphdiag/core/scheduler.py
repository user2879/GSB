"""Budgeted, concurrency-capped query traversal scheduler.

Two independent constraints are enforced:

* ``max_workers`` - global concurrency ceiling for traversal worker threads.
* ``budget`` - one shared :class:`TraversalBudget` counted in visited nodes.
  Every single node expansion must acquire a token; when the budget is
  exhausted no query can continue, remaining queries queue until the session
  is replenished, and the wait is timed on a monotonic clock in milliseconds.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .fixedpoint import Quantizer
from .models import Graph, Query
from .traversal import TraversalResult, run_traversal


class TraversalBudget:
    """Shared visited-node budget with explicit acquire/replenish semantics."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("budget limit must be non-negative")
        self.limit = limit
        self._remaining = limit
        self._consumed = 0
        self._lock = threading.Lock()
        self._refill = threading.Condition(self._lock)

    def acquire(self) -> bool:
        """Consume one visited-node token; ``False`` when exhausted."""
        with self._lock:
            if self._remaining > 0:
                self._remaining -= 1
                self._consumed += 1
                return True
            return False

    def remaining(self) -> int:
        with self._lock:
            return self._remaining

    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def replenish(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("replenish amount must be non-negative")
        with self._refill:
            self._remaining += amount
            self.limit = self.limit + amount
            self._refill.notify_all()

    def wait_for_tokens(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._refill:
            while self._remaining == 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._refill.wait(timeout=remaining)
            return True


@dataclass
class QuerySchedule:
    query_id: str
    enqueue_monotonic_ms: int
    wait_start_monotonic_ms: int | None = None
    wait_end_monotonic_ms: int | None = None
    start_monotonic_ms: int | None = None
    end_monotonic_ms: int | None = None
    visited_nodes: int = 0
    pruned_branches: int = 0
    budget_consumed: int = 0
    budget_exhausted: bool = False
    result: TraversalResult | None = None

    @property
    def wait_duration_ms(self) -> int | None:
        if self.wait_start_monotonic_ms is None or self.wait_end_monotonic_ms is None:
            return None
        return self.wait_end_monotonic_ms - self.wait_start_monotonic_ms


def _now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ScheduleReport:
    schedules: dict[str, QuerySchedule] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def ordered(self) -> list[QuerySchedule]:
        return [self.schedules[qid] for qid in self.order]


def schedule_all(
    graph: Graph,
    queries: tuple[Query, ...],
    quantizer: Quantizer,
    budget: TraversalBudget,
    max_workers: int = 2,
    error_mode: str = "exact",
    wait_timeout: float | None = None,
) -> ScheduleReport:
    """Run all queries under concurrency and budget constraints.

    When the shared budget is exhausted, running traversals stop with
    ``budget_exhausted`` and queued workers block on a monotonic-clock wait.
    If ``wait_timeout`` elapses without replenishment those workers also stop.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    report = ScheduleReport()
    lock = threading.Lock()
    start_gate = threading.Event()

    def worker(query: Query) -> tuple[str, TraversalResult, QuerySchedule]:
        schedule = QuerySchedule(
            query_id=query.query_id, enqueue_monotonic_ms=_now_ms()
        )
        # Deterministic handoff: every query is enqueued before any worker
        # may expand a node, giving an explicit reproducible start boundary.
        start_gate.wait()
        schedule.start_monotonic_ms = _now_ms()
        if budget.remaining() == 0:
            # The session budget is already exhausted: record the mandatory
            # monotonic-clock wait before attempting the first acquisition.
            schedule.wait_start_monotonic_ms = _now_ms()
            got_tokens = budget.wait_for_tokens(timeout=wait_timeout)
            schedule.wait_end_monotonic_ms = _now_ms()
            if not got_tokens:
                schedule.budget_exhausted = True
                result = run_traversal(
                    graph, query, quantizer, _DeniedBudget(), error_mode
                )
                result.budget_exhausted = True
                schedule.result = result
                return query.query_id, result, schedule
        result = _run_with_budget_wait(
            graph, query, quantizer, budget, error_mode, schedule, wait_timeout
        )
        schedule.end_monotonic_ms = _now_ms()
        schedule.visited_nodes = len(result.visited_sequence)
        schedule.pruned_branches = len(result.pruned)
        schedule.budget_consumed = result.budget_visited
        schedule.budget_exhausted = result.budget_exhausted
        schedule.result = result
        return query.query_id, result, schedule

    # All worker threads block on this gate until every query is submitted;
    # the gate is set first so a reused/immediately scheduled thread still
    # observes the released state (setting an already-set event is a no-op).
    start_gate.set()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, query) for query in queries]
        for future in as_completed(futures):
            query_id, _, schedule = future.result()
            with lock:
                report.schedules[query_id] = schedule
                report.order.append(query_id)

    # Present schedules in declaration order (stable), not completion order.
    report.order = [q.query_id for q in queries]
    return report

def _run_with_budget_wait(
    graph, query, quantizer, budget, error_mode, schedule, wait_timeout
) -> TraversalResult:
    """Run a traversal, timing mid-flight waits when the budget is exhausted."""
    # Wrap the shared budget so the first denial starts a timed wait. On
    # replenishment acquisition proceeds normally; on timeout traversal stops.
    waiting = {"active": False}

    class TimedBudget:
        def acquire(self_inner) -> bool:
            while True:
                if budget.acquire():
                    if waiting["active"]:
                        schedule.wait_end_monotonic_ms = _now_ms()
                        waiting["active"] = False
                    return True
                if not waiting["active"]:
                    schedule.wait_start_monotonic_ms = _now_ms()
                    waiting["active"] = True
                if not budget.wait_for_tokens(timeout=wait_timeout):
                    schedule.wait_end_monotonic_ms = _now_ms()
                    waiting["active"] = False
                    return False
                # Signalled: loop and retry the acquisition.

    return run_traversal(graph, query, quantizer, TimedBudget(), error_mode)


class _DeniedBudget:
    """Budget facade that always denies acquisition (wait timed out)."""

    def acquire(self) -> bool:
        return False

"""Bounded parallel call scheduler.

Acquire/renew calls are dispatched to a worker thread pool under two
independent limits:

* a global semaphore caps total in-flight calls;
* a per-holder semaphore caps one holder's in-flight calls.

When quota is exhausted the call queues on the semaphores; the monotonic
wait start/end timestamps are recorded on every attempt. Failures are retried
with the declared integer-millisecond back-off sequence. Retries resend the
exact same (seq, holder, action) request, so single-call semantics are
unchanged (the arbiter treats duplicate seq as idempotent re-delivery).
Exhausting the retry budget terminates the event with ``E_EXHAUST``.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .clock import Clock, MonotonicClock
from .errors import DistLockError, ExhaustError


@dataclass
class CallAttempt:
    attempt: int
    waited_for_quota_ms: int
    started_ms: int
    ended_ms: int
    outcome: str  # OK | RETRY | FAIL
    detail: str = ""


@dataclass
class ScheduledResult:
    seq: int
    holder_id: str
    domain_id: str
    ok: bool
    receipt: dict[str, Any] | None = None
    attempts: list[CallAttempt] = field(default_factory=list)
    error_code: str | None = None
    total_wait_quota_ms: int = 0


class BoundedScheduler:
    def __init__(self, global_limit: int, per_holder_limit: int,
                 backoff_ms: list[int], max_workers: int | None = None,
                 clock: Clock | None = None):
        if global_limit < 1 or per_holder_limit < 1:
            raise ValueError("concurrency limits must be >= 1")
        self.global_limit = global_limit
        self.per_holder_limit = per_holder_limit
        self.backoff_ms = list(backoff_ms)
        self.clock: Clock = clock or MonotonicClock()
        self._global = threading.BoundedSemaphore(global_limit)
        self._holder_locks_guard = threading.Lock()
        self._holder_sems: dict[str, threading.BoundedSemaphore] = {}
        self._inflight_total = 0
        self._inflight_holder: dict[str, int] = {}
        self._peak_total = 0
        self._peak_holder: dict[str, int] = {}
        self._metrics_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers or max(2, global_limit * 2),
            thread_name_prefix="distlock-call")

    def _holder_sem(self, holder_id: str) -> threading.BoundedSemaphore:
        with self._holder_locks_guard:
            sem = self._holder_sems.get(holder_id)
            if sem is None:
                sem = threading.BoundedSemaphore(self.per_holder_limit)
                self._holder_sems[holder_id] = sem
            return sem

    def submit(self, seq: int, holder_id: str, domain_id: str,
               call: Callable[[], dict[str, Any]]) -> ScheduledResult:
        return self.pool.submit(self._run, seq, holder_id, domain_id, call).result()

    def _run(self, seq: int, holder_id: str, domain_id: str,
             call: Callable[[], dict[str, Any]]) -> ScheduledResult:
        result = ScheduledResult(seq=seq, holder_id=holder_id,
                                 domain_id=domain_id, ok=False)
        # Number of transmission attempts = 1 + len(backoff).
        for attempt in range(len(self.backoff_ms) + 1):
            wait_start = self.clock.now_ms()
            holder_sem = self._holder_sem(holder_id)
            # Acquire both quotas; blocking here is the documented queueing.
            self._global.acquire()
            try:
                holder_sem.acquire()
                wait_end = self.clock.now_ms()
                waited = wait_end - wait_start
                entry = CallAttempt(attempt=attempt + 1,
                                    waited_for_quota_ms=waited,
                                    started_ms=wait_end, ended_ms=0,
                                    outcome="OK")
                self._enter_inflight(holder_id)
                try:
                    receipt = call()
                    entry.ended_ms = self.clock.now_ms()
                    result.attempts.append(entry)
                    result.total_wait_quota_ms += waited
                    result.ok = True
                    result.receipt = receipt
                    return result
                except DistLockError as exc:
                    entry.ended_ms = self.clock.now_ms()
                    entry.outcome = "RETRY"
                    entry.detail = exc.code
                    result.attempts.append(entry)
                    result.total_wait_quota_ms += waited
                    last_error = exc.code
                except Exception as exc:  # transport faults are retryable
                    entry.ended_ms = self.clock.now_ms()
                    entry.outcome = "RETRY"
                    entry.detail = type(exc).__name__
                    result.attempts.append(entry)
                    result.total_wait_quota_ms += waited
                    last_error = "E_TRANSPORT"
                finally:
                    self._exit_inflight(holder_id)
                    holder_sem.release()
            finally:
                self._global.release()

            if attempt < len(self.backoff_ms):
                self.clock.sleep_ms(self.backoff_ms[attempt])

        result.ok = False
        result.error_code = "E_EXHAUST"
        raise ExhaustError(
            "retry budget exhausted for event",
            {"seq": seq, "holder_id": holder_id, "domain_id": domain_id,
             "attempts": len(result.attempts), "last_error": last_error},
        )

    def _enter_inflight(self, holder_id: str) -> None:
        with self._metrics_lock:
            self._inflight_total += 1
            self._inflight_holder[holder_id] = self._inflight_holder.get(holder_id, 0) + 1
            self._peak_total = max(self._peak_total, self._inflight_total)
            self._peak_holder[holder_id] = max(
                self._peak_holder.get(holder_id, 0),
                self._inflight_holder[holder_id])

    def _exit_inflight(self, holder_id: str) -> None:
        with self._metrics_lock:
            self._inflight_total -= 1
            self._inflight_holder[holder_id] -= 1

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "global_limit": self.global_limit,
                "per_holder_limit": self.per_holder_limit,
                "peak_inflight_total": self._peak_total,
                "peak_inflight_per_holder": dict(self._peak_holder),
                "current_inflight_total": self._inflight_total,
            }

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)

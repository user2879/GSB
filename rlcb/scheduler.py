"""Three-constraint parallel scheduler projected onto a real thread pool.

Every admitted request must simultaneously hold:

1. one global active-lease permit (``max_active_leases``);
2. one per-service concurrency permit (service ``concurrency_limit``);
3. one global batch-budget permit (``max_active_batches``).

Permits are acquired in a fixed global -> batch -> service order to avoid
deadlock. Requests that would exceed a limit queue and record monotonic-clock
wait start/end milliseconds. The pool genuinely runs work on worker threads;
no code path bypasses the quota gates.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .clock import MonotonicClock


@dataclass
class WaitRecord:
    request_id: str
    service_id: str
    batch_id: str
    wait_start_ms: int
    wait_end_ms: int
    waited_ms: int
    acquired: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_id": self.service_id,
            "batch_id": self.batch_id,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "waited_ms": self.waited_ms,
            "acquired": self.acquired,
        }


class _Counting:
    def __init__(self, limit: int, name: str):
        self.limit = limit
        self.name = name
        self._cond = threading.Condition()
        self._value = 0

    def acquire(self) -> None:
        with self._cond:
            while self._value >= self.limit:
                self._cond.wait()
            self._value += 1

    def release(self) -> None:
        with self._cond:
            self._value -= 1
            self._cond.notify_all()

    @property
    def value(self) -> int:
        with self._cond:
            return self._value


class Scheduler:
    def __init__(self, *, clock: MonotonicClock, max_active_leases: int,
                 max_active_batches: int,
                 service_concurrency: dict[str, int],
                 max_workers: int | None = None):
        self.clock = clock
        self.global_leases = _Counting(max_active_leases, "global_leases")
        self.global_batches = _Counting(max_active_batches, "global_batches")
        self.service_gates = {
            sid: _Counting(max(1, limit), f"service:{sid}")
            for sid, limit in service_concurrency.items()
        }
        self.waits: list[WaitRecord] = []
        self._wait_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers or max(2, max_active_leases + 2),
            thread_name_prefix="rlcb-worker",
        )

    def run(self, tasks: list[dict[str, Any]],
            work: Callable[[dict[str, Any]], Any]) -> list[Any]:
        futures = []
        # Stable submission order; results are re-ordered by request key.
        for task in sorted(tasks, key=lambda t: t["_order"]):
            futures.append(self.executor.submit(self._guarded, task, work))
        results = [future.result() for future in futures]
        return sorted(results, key=lambda r: r["_order"])

    def _guarded(self, task: dict[str, Any],
                 work: Callable[[dict[str, Any]], Any]) -> Any:
        sid = task["service_id"]
        gate = self.service_gates[sid]
        wait_start = self.clock.now_ms()
        # Fixed acquisition order to prevent circular waiting.
        self.global_leases.acquire()
        try:
            self.global_batches.acquire()
            try:
                gate.acquire()
                try:
                    wait_end = self.clock.now_ms()
                    with self._wait_lock:
                        self.waits.append(WaitRecord(
                            request_id=task.get("request_id", task["batch_id"]),
                            service_id=sid,
                            batch_id=task["batch_id"],
                            wait_start_ms=wait_start,
                            wait_end_ms=wait_end,
                            waited_ms=wait_end - wait_start,
                            acquired=True,
                        ))
                    result = work(task)
                    if isinstance(result, dict):
                        result["_order"] = task["_order"]
                    return result
                finally:
                    gate.release()
            finally:
                self.global_batches.release()
        finally:
            self.global_leases.release()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

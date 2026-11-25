"""Quota-aware parallel scheduling on a real thread pool."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .errors import LeaseMiddlewareError
from .timeutil import monotonic_ms


class QuotaExceededError(LeaseMiddlewareError):
    code = "E_QUOTA"


@dataclass
class _Job:
    request_id: str
    node_id: str
    fn: Callable[[], Any]
    future: Future[Any]
    enqueued_ms: int


class QuotaScheduler:
    """Admit work only when global and per-node in-flight quotas allow it.

    The dispatcher admits the FIFO head only.  That deliberately avoids a
    later request bypassing an earlier request whose node quota is saturated.
    Admitted work is projected into a real :class:`ThreadPoolExecutor`.
    """

    def __init__(
        self,
        *,
        global_in_flight: int,
        node_in_flight: dict[str, int],
        thread_name_prefix: str = "lease-worker",
    ) -> None:
        if not isinstance(global_in_flight, int) or global_in_flight <= 0:
            raise ValueError("global_in_flight must be a positive integer")
        self.global_limit = global_in_flight
        self.node_limits = dict(node_in_flight)
        self._queue: deque[_Job] = deque()
        self._condition = threading.Condition()
        self._global_active = 0
        self._node_active: dict[str, int] = {node: 0 for node in node_in_flight}
        self._global_max_seen = 0
        self._node_max_seen: dict[str, int] = {node: 0 for node in node_in_flight}
        self.wait_records: list[dict[str, Any]] = []
        self._executor = ThreadPoolExecutor(
            max_workers=global_in_flight,
            thread_name_prefix=thread_name_prefix,
        )
        self._stopped = False
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="quota-dispatcher", daemon=True
        )
        self._dispatcher.start()

    def submit(self, request_id: str, node_id: str, fn: Callable[[], Any]) -> Future[Any]:
        if node_id not in self.node_limits:
            raise QuotaExceededError(
                "unknown node for scheduler quota",
                field="node_id",
                details={"node_id": node_id},
            )
        future: Future[Any] = Future()
        job = _Job(
            request_id=request_id,
            node_id=node_id,
            fn=fn,
            future=future,
            enqueued_ms=monotonic_ms(),
        )
        with self._condition:
            if self._stopped:
                raise QuotaExceededError("scheduler is stopped", field="scheduler")
            self._queue.append(job)
            self._condition.notify_all()
        return future

    def _can_admit(self, job: _Job) -> bool:
        node_limit = self.node_limits[job.node_id]
        return (
            self._global_active < self.global_limit
            and self._node_active[job.node_id] < node_limit
        )

    def _dispatch_loop(self) -> None:
        while True:
            with self._condition:
                admissible_index: int | None = None
                while not self._stopped:
                    for index, candidate in enumerate(self._queue):
                        if self._can_admit(candidate):
                            admissible_index = index
                            break
                    if admissible_index is not None:
                        break
                    self._condition.wait(timeout=0.02)
                if self._stopped:
                    return
                job = self._queue[admissible_index]
                del self._queue[admissible_index]
                admitted_ms = monotonic_ms()
                self._global_active += 1
                self._node_active[job.node_id] += 1
                self._global_max_seen = max(self._global_max_seen, self._global_active)
                self._node_max_seen[job.node_id] = max(
                    self._node_max_seen[job.node_id], self._node_active[job.node_id]
                )
                self.wait_records.append(
                    {
                        "request_id": job.request_id,
                        "node_id": job.node_id,
                        "wait_started_monotonic_ms": job.enqueued_ms,
                        "wait_ended_monotonic_ms": admitted_ms,
                        "wait_ms": max(0, admitted_ms - job.enqueued_ms),
                        "global_active_after_admit": self._global_active,
                        "node_active_after_admit": self._node_active[job.node_id],
                    }
                )
                self._condition.notify_all()
            self._executor.submit(self._run_job, job)

    def _run_job(self, job: _Job) -> None:
        try:
            result = job.fn()
            if not job.future.done():
                job.future.set_result(result)
        except BaseException as exc:  # propagate to caller future
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            with self._condition:
                self._global_active -= 1
                self._node_active[job.node_id] -= 1
                self._condition.notify_all()

    def active_snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "global_active": self._global_active,
                "global_limit": self.global_limit,
                "node_active": dict(self._node_active),
                "node_limits": dict(self.node_limits),
                "queued": len(self._queue),
                "global_max_seen": self._global_max_seen,
                "node_max_seen": dict(self._node_max_seen),
            }

    def close(self, *, wait: bool = True) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._dispatcher.join(timeout=2)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "QuotaScheduler":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

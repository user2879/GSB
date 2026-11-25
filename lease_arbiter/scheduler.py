from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ScheduledTask:
    request_id: str
    node_id: str
    global_slot: int
    node_slot: int
    wait_started_ms: int
    wait_ended_ms: int

    @property
    def wait_ms(self) -> int:
        return self.wait_ended_ms - self.wait_started_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "global_slot": self.global_slot,
            "node_slot": self.node_slot,
            "wait_started_ms": self.wait_started_ms,
            "wait_ended_ms": self.wait_ended_ms,
            "wait_ms": self.wait_ms,
        }


class QuotaScheduler:
    """Thread-pool dispatcher with hard global and per-node in-flight limits.

    Capacity is acquired before the task is submitted to the executor.  A task
    that cannot obtain both slots waits on the condition; no network function is
    invoked and no thread-pool task exists until both quotas are available.
    """

    def __init__(
        self,
        global_limit: int,
        node_limits: dict[str, int],
        max_workers: int,
        *,
        clock: Callable[[], int] = monotonic_ms,
    ) -> None:
        if global_limit <= 0 or max_workers <= 0:
            raise ValueError("scheduler limits must be positive")
        if any(limit <= 0 for limit in node_limits.values()):
            raise ValueError("node limits must be positive")
        self.global_limit = global_limit
        self.node_limits = dict(node_limits)
        self._clock = clock
        self._condition = threading.Condition()
        self._free_global = list(range(1, global_limit + 1))
        self._free_node = {
            node_id: list(range(1, self.node_limits[node_id] + 1))
            for node_id in sorted(self.node_limits)
        }
        self._inflight_total = 0
        self._inflight_node = {node_id: 0 for node_id in self.node_limits}
        self._max_inflight_total = 0
        self._max_inflight_node = {node_id: 0 for node_id in self.node_limits}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lease-request")
        self._metrics: list[dict[str, Any]] = []
        self._closed = False

    def submit(
        self,
        node_id: str,
        request_id: str,
        fn: Callable[[], T],
    ) -> tuple[Future[T], ScheduledTask]:
        if node_id not in self.node_limits:
            raise ValueError(f"unknown node_id: {node_id}")
        wait_started_ms = self._clock()
        with self._condition:
            while not (
                self._free_global
                and self._free_node[node_id]
                and self._inflight_total < self.global_limit
                and self._inflight_node[node_id] < self.node_limits[node_id]
            ):
                if self._closed:
                    raise RuntimeError("scheduler is closed")
                self._condition.wait()
            if self._closed:
                raise RuntimeError("scheduler is closed")
            global_slot = self._free_global.pop(0)
            node_slot = self._free_node[node_id].pop(0)
            self._inflight_total += 1
            self._inflight_node[node_id] += 1
            self._max_inflight_total = max(self._max_inflight_total, self._inflight_total)
            self._max_inflight_node[node_id] = max(
                self._max_inflight_node[node_id], self._inflight_node[node_id]
            )
            wait_ended_ms = self._clock()
            task = ScheduledTask(
                request_id=request_id,
                node_id=node_id,
                global_slot=global_slot,
                node_slot=node_slot,
                wait_started_ms=wait_started_ms,
                wait_ended_ms=wait_ended_ms,
            )
            try:
                future = self._executor.submit(self._run_task, task, fn)
            except BaseException:
                self._release_locked(task)
                raise
        return future, task

    def _run_task(self, task: ScheduledTask, fn: Callable[[], T]) -> T:
        error: BaseException | None = None
        try:
            return fn()
        except BaseException as exc:
            error = exc
            raise
        finally:
            with self._condition:
                self._release_locked(task)
                metric = task.to_dict()
                metric["ok"] = error is None
                if error is not None:
                    metric["error_code"] = getattr(error, "code", type(error).__name__)
                self._metrics.append(metric)

    def _release_locked(self, task: ScheduledTask) -> None:
        self._inflight_total -= 1
        self._inflight_node[task.node_id] -= 1
        self._free_global.append(task.global_slot)
        self._free_global.sort()
        self._free_node[task.node_id].append(task.node_slot)
        self._free_node[task.node_id].sort()
        self._condition.notify_all()

    def call(self, node_id: str, request_id: str, fn: Callable[[], T]) -> T:
        future, _task = self.submit(node_id, request_id, fn)
        return future.result()

    def metrics(self) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(item) for item in self._metrics]

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "format_version": 1,
                "global_limit": self.global_limit,
                "node_limits": {key: self.node_limits[key] for key in sorted(self.node_limits)},
                "max_inflight_total": self._max_inflight_total,
                "max_inflight_node": {
                    key: self._max_inflight_node[key] for key in sorted(self._max_inflight_node)
                },
                "request_count": len(self._metrics),
                "wait_started_ms": min((item["wait_started_ms"] for item in self._metrics), default=None),
                "wait_ended_ms": max((item["wait_ended_ms"] for item in self._metrics), default=None),
                "requests": [dict(item) for item in self._metrics],
            }

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def __enter__(self) -> "QuotaScheduler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown()

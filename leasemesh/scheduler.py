from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

from .timeutil import monotonic_ms


JobFunction = Callable[[dict[str, Any]], Any]


class BoundedScheduler:
    """Thread-pool scheduler with atomic global and per-node in-flight caps."""

    def __init__(self, global_limit: int, per_node_limit: int) -> None:
        if global_limit <= 0 or per_node_limit <= 0:
            raise ValueError("scheduler limits must be positive")
        self.global_limit = int(global_limit)
        self.per_node_limit = int(per_node_limit)
        self.pool = ThreadPoolExecutor(
            max_workers=self.global_limit,
            thread_name_prefix="leasemesh-client",
        )
        self.condition = threading.Condition()
        self.active_global = 0
        self.active_by_node: dict[str, int] = {}
        self.max_active_global = 0
        self.max_active_by_node: dict[str, int] = {}
        self.submitted = 0
        self.completed = 0
        self._closed = False

    def submit(self, request: dict[str, Any], function: JobFunction) -> Future[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("scheduler is shut down")
        with self.condition:
            self.submitted += 1
        job = dict(request)
        job["wait_started_ms"] = monotonic_ms()
        job["queue_epoch"] = self.submitted
        return self.pool.submit(self._run, job, function)

    def _run(self, job: dict[str, Any], function: JobFunction) -> dict[str, Any]:
        node_id = str(job["node_id"])
        with self.condition:
            while (
                self.active_global >= self.global_limit
                or self.active_by_node.get(node_id, 0) >= self.per_node_limit
            ):
                self.condition.wait()
            self.active_global += 1
            self.active_by_node[node_id] = self.active_by_node.get(node_id, 0) + 1
            self.max_active_global = max(self.max_active_global, self.active_global)
            self.max_active_by_node[node_id] = max(
                self.max_active_by_node.get(node_id, 0),
                self.active_by_node[node_id],
            )
            job["wait_ended_ms"] = monotonic_ms()
            job["wait_ms"] = max(0.0, float(job["wait_ended_ms"]) - float(job["wait_started_ms"]))
            job["active_global_at_start"] = self.active_global
            job["active_node_at_start"] = self.active_by_node[node_id]

        try:
            response = function(job)
            return {"request": job, "response": response, "ok": True}
        except Exception as exc:
            job["wait_ended_ms"] = job.get("wait_ended_ms") or monotonic_ms()
            job["wait_ms"] = max(0.0, float(job["wait_ended_ms"]) - float(job["wait_started_ms"]))
            return {"request": job, "error": exc, "ok": False}
        finally:
            with self.condition:
                self.active_global -= 1
                self.active_by_node[node_id] -= 1
                if self.active_by_node[node_id] <= 0:
                    self.active_by_node.pop(node_id, None)
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "active_global": self.active_global,
                "active_by_node": dict(self.active_by_node),
                "max_active_global": self.max_active_global,
                "max_active_by_node": dict(self.max_active_by_node),
                "submitted": self.submitted,
                "completed": self.completed,
                "global_limit": self.global_limit,
                "per_node_limit": self.per_node_limit,
            }

    def shutdown(self, wait: bool = True) -> None:
        self._closed = True
        self.pool.shutdown(wait=wait)

    def __enter__(self) -> "BoundedScheduler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown(wait=True)

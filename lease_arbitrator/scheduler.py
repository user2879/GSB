from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .protocol import monotonic_ms, monotonic_ns


class QuotaGate:
    def __init__(
        self,
        version: int,
        *,
        global_concurrency: int,
        stage_quotas: dict[str, int],
        node_concurrency: dict[str, int],
    ) -> None:
        self.version = version
        self.global_concurrency = global_concurrency
        self.stage_quotas = dict(stage_quotas)
        self.node_concurrency = dict(node_concurrency)
        self.global_slots = threading.BoundedSemaphore(global_concurrency)
        self.stage_slots = {
            stage: threading.BoundedSemaphore(max(1, limit))
            for stage, limit in stage_quotas.items()
        }
        self.node_slots = {
            node: threading.BoundedSemaphore(max(1, limit))
            for node, limit in node_concurrency.items()
        }

    def _stage_semaphore(self, stage: str | None) -> threading.BoundedSemaphore:
        if stage is None:
            return threading.BoundedSemaphore(10**9)
        if stage not in self.stage_slots:
            raise KeyError(f"unknown stage quota {stage!r}")
        return self.stage_slots[stage]

    def _node_semaphore(self, node_id: str | None) -> threading.BoundedSemaphore:
        if node_id is None:
            return threading.BoundedSemaphore(10**9)
        if node_id not in self.node_slots:
            raise KeyError(f"unknown node concurrency {node_id!r}")
        return self.node_slots[node_id]

    def run(
        self,
        *,
        event_seq: int,
        stage: str | None,
        node_id: str | None,
        fn: Callable[[], Any],
    ) -> tuple[Any, dict[str, Any]]:
        stage_slots = self._stage_semaphore(stage)
        node_slots = self._node_semaphore(node_id)
        semaphores = [self.global_slots, stage_slots, node_slots]
        wait_start_ns = monotonic_ns()
        wait_start_ms = monotonic_ms()

        acquired: list[threading.BoundedSemaphore] = []
        exhausted = False
        for semaphore in semaphores:
            if semaphore.acquire(blocking=False):
                acquired.append(semaphore)
            else:
                exhausted = True
                break

        if exhausted:
            for semaphore in reversed(acquired):
                semaphore.release()
            acquired.clear()
            for semaphore in semaphores:
                semaphore.acquire()
                acquired.append(semaphore)

        wait_end_ns = monotonic_ns()
        wait_end_ms = monotonic_ms()
        metric = {
            "event_seq": event_seq,
            "stage": stage,
            "node_id": node_id,
            "quota_version": self.version,
            "queued": exhausted,
            "wait_start_monotonic_ms": wait_start_ms,
            "wait_end_monotonic_ms": wait_end_ms,
            "wait_ms": max(0, (wait_end_ns - wait_start_ns) // 1_000_000),
            "global_concurrency": self.global_concurrency,
        }
        try:
            value = fn()
        except Exception as exc:
            setattr(exc, "queue_metric", metric)
            raise
        finally:
            for semaphore in reversed(acquired):
                semaphore.release()
        return value, metric


class DeliveryScheduler:
    def __init__(
        self,
        gate: QuotaGate,
        *,
        max_workers: int | None = None,
    ) -> None:
        self.gate = gate
        workers = max_workers or max(4, gate.global_concurrency * 2)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cache-delivery")
        self._lock = threading.RLock()

    def replace_gate(self, gate: QuotaGate) -> None:
        with self._lock:
            self.gate = gate

    def run(
        self,
        *,
        event_seq: int,
        stage: str | None,
        node_id: str | None,
        fn: Callable[[], Any],
    ) -> tuple[Any, dict[str, Any]]:
        with self._lock:
            gate = self.gate
        return gate.run(event_seq=event_seq, stage=stage, node_id=node_id, fn=fn)

    def parallel(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run node deliveries on a real thread pool.

        Each task has node_id, stage and fn. Results are returned in the
        caller's task order, while execution may occur in parallel.
        """
        futures = []
        for task in tasks:
            futures.append(
                self.executor.submit(
                    self.run,
                    event_seq=task["event_seq"],
                    stage=task.get("stage"),
                    node_id=task.get("node_id"),
                    fn=task["fn"],
                )
            )
        results: list[dict[str, Any]] = []
        for task, future in zip(tasks, futures):
            try:
                value, metric = future.result()
                results.append({"ok": True, "task": task, "value": value, "queue": metric})
            except Exception as exc:
                if hasattr(exc, "to_dict"):
                    error = exc.to_dict()
                else:
                    error = {"code": getattr(exc, "code", "E_DELIVERY"), "message": str(exc)}
                results.append({"ok": False, "task": task, "error": error, "queue": getattr(exc, "queue_metric", None)})
        return results

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=not wait)

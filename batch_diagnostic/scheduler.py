"""Parallel receipt processing under global and per-worker concurrency caps."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .network import WorkerClient
from .spec import SystemSpec


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class BatchTiming:
    batch_id: str
    wait_started_ms: int
    wait_ended_ms: int
    elapsed_ms: int
    attempts: int

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "wait_started_ms": self.wait_started_ms,
            "wait_ended_ms": self.wait_ended_ms,
            "elapsed_ms": self.elapsed_ms,
            "attempts": self.attempts,
        }


@dataclass
class ScheduledResult:
    batch_id: str
    worker_id: str
    response: dict
    timing: BatchTiming


@dataclass
class Scheduler:
    spec: SystemSpec
    client_factory: Callable[[str, int], WorkerClient]
    global_semaphore: threading.Semaphore = field(init=False)
    worker_semaphores: dict[str, threading.Semaphore] = field(init=False)
    peak_global: int = 0
    peak_worker: dict[str, int] = field(default_factory=dict)
    _counters_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.global_semaphore = threading.Semaphore(self.spec.budget.global_max_in_flight)
        self.worker_semaphores = {
            worker.worker_id: threading.Semaphore(worker.max_in_flight)
            for worker in self.spec.workers
        }
        self._active_global = 0
        self._active_worker = {worker.worker_id: 0 for worker in self.spec.workers}

    def _acquire(self, worker_id: str, wait_started_ms: int) -> int:
        self.global_semaphore.acquire()
        self.worker_semaphores[worker_id].acquire()
        wait_ended_ms = now_ms()
        with self._counters_lock:
            self._active_global += 1
            self._active_worker[worker_id] += 1
            self.peak_global = max(self.peak_global, self._active_global)
            self.peak_worker[worker_id] = max(
                self.peak_worker.get(worker_id, 0), self._active_worker[worker_id]
            )
        return wait_ended_ms

    def _release(self, worker_id: str) -> None:
        with self._counters_lock:
            self._active_global -= 1
            self._active_worker[worker_id] -= 1
        self.worker_semaphores[worker_id].release()
        self.global_semaphore.release()

    def _process(self, job: dict) -> ScheduledResult:
        worker_id = job["worker_id"]
        worker = self.spec.worker_by_id()[worker_id]
        wait_started_ms = now_ms()
        wait_ended_ms = self._acquire(worker_id, wait_started_ms)
        started_ms = now_ms()
        try:
            client = self.client_factory(worker.host, worker.port)
            response, attempts = client.round_trip_with_retry(job, worker.retries)
            elapsed_ms = now_ms() - started_ms
            timing = BatchTiming(
                batch_id=job["batch_id"],
                wait_started_ms=wait_started_ms,
                wait_ended_ms=wait_ended_ms,
                elapsed_ms=elapsed_ms,
                attempts=attempts,
            )
            return ScheduledResult(job["batch_id"], worker_id, response, timing)
        finally:
            self._release(worker_id)

    def run(self, jobs: list[dict]) -> list[ScheduledResult]:
        """Project jobs onto a real thread pool and return stable-order results."""

        ordered = sorted(jobs, key=lambda job: (job["worker_id"], job["batch_index"]))
        results: list[ScheduledResult] = []
        with ThreadPoolExecutor(max_workers=self.spec.budget.global_max_in_flight) as pool:
            futures = [pool.submit(self._process, job) for job in ordered]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: (result.worker_id, result.response["batch_index"]))
        return results

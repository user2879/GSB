"""Quota-constrained parallel batch scheduler.

Enforces a global in-flight limit and a per-worker quota.  Batches that cannot
start immediately queue, and their wait window is recorded with
``time.monotonic_ns`` converted to integer milliseconds.  Wall-clock timing
never feeds content hashes, so scheduling nondeterminism cannot change the
logical ledger content.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue
from .batching import Batch


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class BatchRun:
    batch: Batch
    wait_start_ms: int
    wait_end_ms: int
    run_start_ms: int
    run_end_ms: int


class Scheduler:
    def __init__(self, max_global: int, per_worker_quota: int):
        if max_global <= 0 or per_worker_quota <= 0:
            raise ValueError("concurrency limits must be positive")
        self.max_global = max_global
        self.per_worker_quota = per_worker_quota
        self._global = threading.Semaphore(max_global)
        self._worker_locks: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()

    def _worker_semaphore(self, worker_id: str) -> threading.Semaphore:
        with self._lock:
            if worker_id not in self._worker_locks:
                self._worker_locks[worker_id] = threading.Semaphore(
                    self.per_worker_quota)
            return self._worker_locks[worker_id]

    def run(self, batches: tuple[Batch, ...], worker) -> list[BatchRun]:
        ordered = sorted(batches, key=lambda b: (b.worker_id,
                                                 b.param_version_seq,
                                                 b.batch_index))
        results: list[BatchRun] = []
        results_lock = threading.Lock()
        queue: Queue = Queue()
        for batch in ordered:
            queue.put(batch)

        def loop():
            while True:
                batch = queue.get()
                try:
                    if batch is None:
                        return
                    wait_start = now_ms()
                    worker_sem = self._worker_semaphore(batch.worker_id)
                    self._global.acquire()
                    worker_sem.acquire()
                    wait_end = now_ms()
                    run_start = now_ms()
                    try:
                        worker(batch)
                    finally:
                        run_end = now_ms()
                        worker_sem.release()
                        self._global.release()
                    with results_lock:
                        results.append(BatchRun(
                            batch=batch,
                            wait_start_ms=wait_start,
                            wait_end_ms=wait_end,
                            run_start_ms=run_start,
                            run_end_ms=run_end,
                                ))
                finally:
                    queue.task_done()

        threads = [threading.Thread(target=loop, daemon=True)
                   for _ in range(min(self.max_global, len(ordered) or 1))]
        for thread in threads:
            thread.start()
        queue.join()
        for _ in threads:
            queue.put(None)
        for thread in threads:
            thread.join()
        return sorted(results, key=lambda r: (r.batch.worker_id,
                                              r.batch.param_version_seq,
                                              r.batch.batch_index))

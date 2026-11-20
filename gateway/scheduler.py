"""Parallel batch scheduler under dual budgets.

A global concurrency cap and a per-backend quota are both enforced with
semaphores; queued batches record wait start/end on a monotonic millisecond
clock. Each batch really executes the local NumPy forward pass and records an
integer duration. Timeout re-dispatch never changes the batch input set or
the output digest.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .forward import batch_digest, forward_scores, summarize
from .models import Batch, BatchResult, Declaration

MAX_ATTEMPTS = 3


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


class Scheduler:
    def __init__(self, decl: Declaration, global_limit: int,
                 forward_repeat: int = 1, timeout_ms: Optional[int] = None,
                 on_batch_committed: Optional[Callable[[BatchResult], None]] = None,
                 crash_after_commits: Optional[int] = None):
        self.decl = decl
        self.global_sem = threading.BoundedSemaphore(global_limit)
        self.backend_sems = {
            b.model_id: threading.BoundedSemaphore(b.max_concurrency)
            for b in decl.backends
        }
        self.params = {b.model_id: b.params for b in decl.backends}
        self.forward_repeat = forward_repeat
        self.timeout_ms = timeout_ms
        self.on_batch_committed = on_batch_committed
        self.crash_after_commits = crash_after_commits
        self._commits = 0
        self._lock = threading.Lock()
        self._peak_running = 0
        self._running = 0
        self._backend_peak: Dict[str, int] = {b.model_id: 0 for b in decl.backends}
        self._backend_running: Dict[str, int] = {b.model_id: 0 for b in decl.backends}

    @property
    def peak_running(self) -> int:
        return self._peak_running

    @property
    def backend_peak(self) -> Dict[str, int]:
        return dict(self._backend_peak)

    def _execute_once(self, batch: Batch) -> Tuple[Dict[int, int], str, int, int]:
        inputs = tuple(self.decl.requests[seq].inputs for seq in batch.request_seqs)
        start = _now_ms()
        scores = forward_scores(self.params[batch.backend_id], inputs,
                                repeat=self.forward_repeat)
        end = _now_ms()
        summaries = summarize(scores)
        return summaries, batch_digest(summaries), start, end

    def _run_batch(self, batch: Batch, fingerprint: str) -> BatchResult:
        wait_start = _now_ms()
        self.global_sem.acquire()
        self.backend_sems[batch.backend_id].acquire()
        wait_end = _now_ms()
        with self._lock:
            self._running += 1
            self._backend_running[batch.backend_id] += 1
            self._peak_running = max(self._peak_running, self._running)
            self._backend_peak[batch.backend_id] = max(
                self._backend_peak[batch.backend_id],
                self._backend_running[batch.backend_id])
        try:
            attempt = 0
            while True:
                attempt += 1
                summaries, digest, start, end = self._execute_once(batch)
                timed_out = (self.timeout_ms is not None
                             and end - start >= self.timeout_ms
                             and attempt < MAX_ATTEMPTS)
                if not timed_out:
                    break
                # Timeout re-dispatch: identical input set, identical digest.
            result = BatchResult(
                batch=batch, fingerprint=fingerprint,
                wait_start_ms=wait_start, wait_end_ms=wait_end,
                start_ms=start, end_ms=end, duration_ms=end - start,
                output_digest=digest, attempt=attempt)
            if self.on_batch_committed is not None:
                self.on_batch_committed(result)
            with self._lock:
                self._commits += 1
                if (self.crash_after_commits is not None
                        and self._commits >= self.crash_after_commits):
                    import os
                    os._exit(137)  # simulated forced termination
            return result
        finally:
            with self._lock:
                self._running -= 1
                self._backend_running[batch.backend_id] -= 1
            self.backend_sems[batch.backend_id].release()
            self.global_sem.release()

    def run(self, batches: List[Batch],
            fingerprints: Dict[str, str]) -> List[BatchResult]:
        results: List[BatchResult] = []
        lock = threading.Lock()

        def worker(batch: Batch) -> None:
            result = self._run_batch(batch, fingerprints[batch.backend_id])
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker, args=(b,)) for b in batches]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # Stable result order: canonical batch order, not completion order.
        order = {b.batch_id: i for i, b in enumerate(batches)}
        results.sort(key=lambda r: order[r.batch.batch_id])
        return results

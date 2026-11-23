"""Parallel replay/renewal scheduling under global and per-shard quotas.

Tasks run on a real worker thread pool. The global running-task count never
exceeds max_workers and the per-shard running count never exceeds the
per-shard quota; tasks that cannot start queue, and their wait start/end
monotonic-millisecond window is recorded.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

from .clock import mono_ms


class Scheduler:
    def __init__(self, max_workers, per_shard_quota, logger=None):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self.per_shard_quota = max(1, per_shard_quota)
        self.logger = logger
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="kvdiag")
        self._shard_sem = {}
        self._lock = threading.Lock()
        self.wait_events = []
        self._running_global = 0
        self._max_running_global = 0
        self._running_shard = {}
        self._max_running_shard = {}

    def _semaphore(self, shard_id):
        with self._lock:
            sem = self._shard_sem.get(shard_id)
            if sem is None:
                sem = threading.Semaphore(self.per_shard_quota)
                self._shard_sem[shard_id] = sem
            return sem

    def submit(self, shard_id, task_kind, fn, *args, **kwargs):
        sem = self._semaphore(shard_id)
        enqueue_ms = mono_ms()

        def guarded():
            start = enqueue_ms
            sem.acquire()
            end = mono_ms()
            event = {"shard_id": shard_id, "task_kind": task_kind,
                     "wait_start_mono_ms": start, "wait_end_mono_ms": end}
            with self._lock:
                self.wait_events.append(event)
                self._running_global += 1
                self._max_running_global = max(self._max_running_global,
                                               self._running_global)
                self._running_shard[shard_id] = self._running_shard.get(shard_id, 0) + 1
                self._max_running_shard[shard_id] = max(
                    self._max_running_shard.get(shard_id, 0),
                    self._running_shard[shard_id])
            if self.logger:
                self.logger.emit("task_started", shard_id=shard_id,
                                 task_kind=task_kind,
                                 wait_start_mono_ms=start, wait_end_mono_ms=end)
            try:
                return fn(*args, **kwargs)
            finally:
                sem.release()
                with self._lock:
                    self._running_global -= 1
                    self._running_shard[shard_id] -= 1

        return self._pool.submit(guarded)

    def shutdown(self):
        self._pool.shutdown(wait=True)

    def stats(self):
        with self._lock:
            return {
                "max_running_global": self._max_running_global,
                "max_running_shard": dict(self._max_running_shard),
                "wait_events": list(self.wait_events),
            }

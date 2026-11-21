"""Probe scheduler under a global concurrency budget, per-pool quotas and
a global rate limit.

A single dispatcher thread walks the deterministic task list in order:
for each task it (1) waits for a rate-limit token, (2) acquires the global
budget semaphore and the pool quota semaphore, (3) records the monotonic
queue wait and (4) hands the probe to a worker thread. Because dispatch is
single-threaded and the task list is sorted, the initiated instance
sequence is identical across runs of the same declaration.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, rate_per_sec):
        self.rate = rate_per_sec
        self._starts = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= 1.0:
                    self._starts.popleft()
                if len(self._starts) < self.rate:
                    self._starts.append(now)
                    return
                sleep_for = 1.0 - (now - self._starts[0])
            time.sleep(max(sleep_for, 0.001))


class Scheduler:
    def __init__(self, global_budget, rate_per_sec, pool_quotas):
        self._global_sem = threading.BoundedSemaphore(global_budget)
        self._rate = RateLimiter(rate_per_sec)
        self._pool_sems = {
            pid: threading.BoundedSemaphore(q)
            for pid, q in pool_quotas.items()
        }
        self._stats_lock = threading.Lock()
        self.in_flight_global = 0
        self.in_flight_pool = {pid: 0 for pid in pool_quotas}
        self.peak_global = 0
        self.peak_pool = {pid: 0 for pid in pool_quotas}

    def _enter(self, pool_id):
        with self._stats_lock:
            self.in_flight_global += 1
            self.in_flight_pool[pool_id] += 1
            self.peak_global = max(self.peak_global, self.in_flight_global)
            self.peak_pool[pool_id] = max(self.peak_pool[pool_id],
                                          self.in_flight_pool[pool_id])

    def _exit(self, pool_id):
        with self._stats_lock:
            self.in_flight_global -= 1
            self.in_flight_pool[pool_id] -= 1
        self._pool_sems[pool_id].release()
        self._global_sem.release()

    def run_round(self, tasks, probe_fn, on_result):
        """Dispatch tasks in list order; block when budget/quota/rate deny.

        ``tasks``     : list of dicts (pool_id, instance_id, seq, round,
                        path, plus probe kwargs)
        ``probe_fn``  : callable(task) -> (outcome, reason, detail)
        ``on_result`` : callable(task, outcome, reason, detail) executed by
                        the worker thread once the probe finishes
        Returns the list of dispatch records (instance, pool, wait_ms).
        """
        dispatched = []
        threads = []

        def worker(task):
            try:
                outcome, reason, detail = probe_fn(task)
                task["finished_ms"] = time.monotonic() * 1000.0
                on_result(task, outcome, reason, detail)
            finally:
                self._exit(task["pool_id"])

        for task in tasks:
            task["enqueued_ms"] = time.monotonic() * 1000.0
            self._global_sem.acquire()
            self._pool_sems[task["pool_id"]].acquire()
            # the rate token is the LAST gate before dispatch, so the
            # recorded start times themselves honour the per-second limit
            self._rate.acquire()
            task["started_ms"] = time.monotonic() * 1000.0
            task["wait_ms"] = task["started_ms"] - task["enqueued_ms"]
            self._enter(task["pool_id"])
            dispatched.append({
                "seq": task["seq"],
                "instance_id": task["instance_id"],
                "pool_id": task["pool_id"],
                "wait_ms": round(task["wait_ms"], 3),
            })
            t = threading.Thread(target=worker, args=(task,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return dispatched

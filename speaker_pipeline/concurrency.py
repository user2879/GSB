"""Bounded thread pool with stable sort-key based result ordering.

Thread scheduling is non-deterministic, but every public method returns
results ordered by an explicit stable key derived from file/frame/interval
indexes, so downstream products are identical to serial execution.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class BoundedWorkerPool:
    def __init__(self, max_workers):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = int(max_workers)
        self._active = 0
        self._high_water = 0
        self._lock = threading.Lock()

    @property
    def high_water_mark(self):
        return self._high_water

    def _track(self, fn, item):
        with self._lock:
            self._active += 1
            self._high_water = max(self._high_water, self._active)
        try:
            return fn(item)
        finally:
            with self._lock:
                self._active -= 1

    def map_ordered(self, fn, items, sort_key):
        """Map *fn* over items concurrently, return results in key order."""
        ordered = sorted(items, key=sort_key)
        if self.max_workers == 1 or len(ordered) <= 1:
            return [fn(item) for item in ordered]
        indexed = list(enumerate(ordered))
        results = [None] * len(indexed)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._track, fn, item): position
                for position, item in indexed
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def map_isolated(self, fn, items, sort_key, on_error):
        """Concurrent map where a failing item is recorded, not fatal.

        Returns ``(results, failures)`` both in stable key order.
        """
        ordered = sorted(items, key=sort_key)
        results, failures = [], []

        def guarded(item):
            try:
                return ("ok", fn(item))
            except Exception as exc:  # isolated per-item failure
                return ("err", exc)

        mapped = self.map_ordered(guarded, ordered, sort_key)
        for item, outcome in zip(ordered, mapped):
            status, payload = outcome
            if status == "ok":
                results.append(payload)
            else:
                failures.append(on_error(item, payload))
        return results, failures

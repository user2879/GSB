"""Bounded thread pool built only from threading primitives."""

import queue
import threading


class BoundedThreadPool:
    """A small worker pool whose real concurrency never exceeds a bound."""

    def __init__(self, max_workers):
        self.max_workers = max(1, int(max_workers))
        self._tasks = queue.Queue()
        self._results = []
        self._lock = threading.Lock()
        self._threads = []
        self._sentinel = object()
        for _ in range(self.max_workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _worker(self):
        while True:
            item = self._tasks.get()
            if item is self._sentinel:
                self._tasks.task_done()
                return
            order, func, args, kwargs = item
            try:
                value = func(*args, **kwargs)
                error = None
            except Exception as exc:  # captured, never kills a worker
                value = None
                error = exc
            with self._lock:
                self._results.append((order, value, error))
            self._tasks.task_done()

    def map_ordered(self, jobs):
        """Submit (func, args, kwargs) jobs and return results in input order."""
        for order, (func, args, kwargs) in enumerate(jobs):
            self._tasks.put((order, func, args, kwargs))
        self._tasks.join()
        ordered = sorted(self._results, key=lambda item: item[0])
        self._results = []
        return ordered

    def shutdown(self):
        for _ in self._threads:
            self._tasks.put(self._sentinel)
        for thread in self._threads:
            thread.join()

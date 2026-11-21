from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class ParallelRunner:
    """Bounded executor that returns results in stable-key order."""

    def __init__(self, max_workers: int, chunk_size: int = 64):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="driftbench")
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.submitted = 0
        self._yield_budget = max_workers * 4

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "ParallelRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _run(self, function: Callable[[], Any]) -> Any:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            with self._lock:
                should_yield = self.max_workers > 1 and self._yield_budget > 0
                if should_yield:
                    self._yield_budget -= 1
            if should_yield:
                # One-time startup yield makes the worker cap observable even when
                # deterministic event tasks are only a few microseconds long.
                time.sleep(0.0002)
            return function()
        finally:
            with self._lock:
                self._active -= 1

    def map_tasks(self, tasks: list[tuple[str, Callable[[], Any]]]) -> list[Any]:
        """Run tasks and return results sorted by their stable key."""
        ordered = sorted(tasks, key=lambda item: item[0])
        results: list[tuple[str, Future]] = []
        for offset in range(0, len(ordered), self.chunk_size):
            chunk = ordered[offset : offset + self.chunk_size]
            for key, function in chunk:
                results.append((key, self._executor.submit(self._run, function)))
                with self._lock:
                    self.submitted += 1
        output: list[Any] = []
        first_error: BaseException | None = None
        for key, future in sorted(results, key=lambda item: item[0]):
            try:
                output.append(future.result())
            except BaseException as exc:  # preserve stable-key first error
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return output

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "max_observed_active": self.max_active,
                "submitted_tasks": self.submitted,
            }

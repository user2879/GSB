"""Bounded thread-pool helper with deterministic merge order."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable


class TrackingExecutor:
    def __init__(self, max_workers: int):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mgirecon")
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.submitted = 0

    def __enter__(self) -> "TrackingExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _wrapped(self, function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return function(*args, **kwargs)
        finally:
            with self._lock:
                self._active -= 1

    def map_sorted(
        self,
        items: Iterable[Any],
        key: Callable[[Any], str],
        function: Callable[[Any], Any],
    ) -> list[Any]:
        ordered = sorted(items, key=key)
        futures = [self._executor.submit(self._wrapped, function, (item,), {}) for item in ordered]
        self.submitted += len(futures)
        return [future.result() for future in futures]

    def stats(self) -> dict[str, int]:
        return {
            "max_workers": self.max_workers,
            "max_active": self.max_active,
            "submitted": self.submitted,
        }

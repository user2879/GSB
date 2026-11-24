from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ConcurrencyMonitor:
    limit: int
    active: int = 0
    max_observed: int = 0
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def enter(self) -> None:
        assert self._lock is not None
        with self._lock:
            self.active += 1
            self.max_observed = max(self.max_observed, self.active)
            if self.active > self.limit:
                raise RuntimeError(f"concurrency limit exceeded: {self.active} > {self.limit}")

    def exit(self) -> None:
        assert self._lock is not None
        with self._lock:
            self.active -= 1


def stable_task_key(batch: int, seq: int, stage_name: str, stage_rank: int = 0) -> tuple[int, int, int, str]:
    """Canonical key. Stage rank keeps dependency order; all fields are explicit."""

    return (batch, seq, stage_rank, stage_name)


def parallel_map_ordered(
    items: Sequence[T],
    worker: Callable[[T], R],
    max_workers: int,
    monitor: ConcurrencyMonitor | None = None,
) -> list[R]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    def guarded(item: T) -> R:
        if monitor is not None:
            monitor.enter()
        try:
            return worker(item)
        finally:
            if monitor is not None:
                monitor.exit()

    if max_workers == 1 or len(items) <= 1:
        return [guarded(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(guarded, item) for item in items]
        return [future.result() for future in futures]

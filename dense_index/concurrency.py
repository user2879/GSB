from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Tuple


class ConcurrencyMonitor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.observed_max = 0

    @contextmanager
    def enter(self):
        with self._lock:
            self.active += 1
            self.observed_max = max(self.observed_max, self.active)
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1


def run_ordered(
    tasks: List[Any],
    worker: Callable[[Any], Any],
    max_workers: int,
    monitor: ConcurrencyMonitor | None = None,
) -> Tuple[List[Any], List[BaseException | None]]:
    """Run tasks concurrently but return outcomes in input order."""

    monitor = monitor or ConcurrencyMonitor()
    results: List[Any] = [None] * len(tasks)
    errors: List[BaseException | None] = [None] * len(tasks)
    if not tasks:
        return results, errors

    def wrapped(index: int, task: Any) -> Tuple[int, Any]:
        with monitor.enter():
            return index, worker(task)

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_to_index = {executor.submit(wrapped, index, task): index for index, task in enumerate(tasks)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                _, value = future.result()
                results[index] = value
            except BaseException as exc:  # captured and re-raised after ordering/cleanup
                errors[index] = exc
    return results, errors

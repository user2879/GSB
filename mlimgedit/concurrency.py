"""Deterministic thread-pool runner.

Tasks carry explicit stable sort keys. Work may execute on a thread pool, but
results are always collected and emitted in stable-key order, which makes
concurrency level and OS scheduling irrelevant to the final artifacts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class OrderedTask(Generic[T, R]):
    key: str
    payload: T
    func: Callable[[T], R]
    stage: str
    classifier: str = ""
    sample: str = ""

    def execute(self) -> R:
        return self.func(self.payload)


def run_tasks(tasks: list[OrderedTask], threads: int) -> list[tuple[OrderedTask, R | Exception]]:
    """Run tasks with at most ``threads`` workers and return key-ordered output."""
    ordered = sorted(tasks, key=lambda task: task.key)
    threads = max(1, min(int(threads), len(ordered) or 1))
    results: list[tuple[OrderedTask, R | Exception]] = []
    if threads == 1 or len(ordered) <= 1:
        for task in ordered:
            try:
                results.append((task, task.execute()))
            except Exception as exc:  # isolation: caller maps this to a failure record
                results.append((task, exc))
        return results

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_map = {executor.submit(task.execute): task for task in ordered}
        collected: list[tuple[OrderedTask, R | Exception]] = []
        for future, task in future_map.items():
            try:
                collected.append((task, future.result()))
            except Exception as exc:
                collected.append((task, exc))
    # Re-sort by the deterministic key regardless of future completion order.
    collected.sort(key=lambda pair: pair[0].key)
    return collected

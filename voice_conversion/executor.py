"""Serial and bounded thread-pool executors with deterministic output order."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_tasks(tasks: Iterable[tuple], worker: Callable[..., R],
              workers: int) -> list[R]:
    """Run ``worker(*task)`` and return results sorted by the stable key.

    Each task tuple starts with ``(segment_index, stage_name, ...payload)``.
    Results are reordered by that key, so thread scheduling cannot change the
    final segment ordering.  The pool size never exceeds the configured limit.
    """
    ordered = sorted(tasks, key=lambda item: (item[0], item[1]))
    if workers <= 1 or len(ordered) <= 1:
        return [worker(*task) for task in ordered]
    with ThreadPoolExecutor(max_workers=min(workers, len(ordered))) as pool:
        futures = [pool.submit(worker, *task) for task in ordered]
        return [future.result() for future in futures]

"""Deterministic ordered task execution with a hard worker limit."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class OrderedTask(Generic[T]):
    sort_key: str
    payload: T


@dataclass(frozen=True)
class StageExecution(Generic[R]):
    results: tuple[R, ...]
    sort_keys: tuple[str, ...]
    max_observed: int
    worker_limit: int


def execute_ordered(
    tasks: Iterable[OrderedTask[T]],
    worker: Callable[[T], R],
    *,
    max_workers: int,
    stage: str,
) -> StageExecution[R]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    ordered = sorted(tasks, key=lambda task: task.sort_key)
    if not ordered:
        return StageExecution((), (), 0, max_workers)
    active = 0
    max_observed = 0
    lock = threading.Lock()

    def guarded(payload: T) -> R:
        nonlocal active, max_observed
        with lock:
            active += 1
            max_observed = max(max_observed, active)
        try:
            return worker(payload)
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"gsb-{stage}") as pool:
        futures = [pool.submit(guarded, task.payload) for task in ordered]
        results = tuple(future.result() for future in futures)
    if max_observed > max_workers:
        raise RuntimeError("worker limit was exceeded")
    return StageExecution(
        results=results,
        sort_keys=tuple(task.sort_key for task in ordered),
        max_observed=max_observed,
        worker_limit=max_workers,
    )

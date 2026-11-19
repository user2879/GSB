from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def stable_task_key(
    instance_id: int,
    stage: str,
    *,
    source_format: str = "",
) -> str:
    return f"{stage}:{source_format}:{int(instance_id):020d}"


@dataclass(frozen=True)
class OrderedTask(Generic[T]):
    instance_id: int
    stage: str
    payload: T
    source_format: str = ""

    @property
    def stable_key(self) -> str:
        return stable_task_key(
            self.instance_id,
            self.stage,
            source_format=self.source_format,
        )

    @property
    def sort_key(self) -> tuple:
        return (
            self.source_format,
            int(self.instance_id),
            self.stage,
            self.stable_key,
        )


def ordered_parallel_map(
    tasks: list[OrderedTask[T]],
    worker: Callable[[T], R],
    max_workers: int,
) -> list[R]:
    ordered = sorted(tasks, key=lambda task: task.sort_key)
    if max_workers <= 1 or len(ordered) <= 1:
        return [worker(task.payload) for task in ordered]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            (task.stable_key, executor.submit(worker, task.payload))
            for task in ordered
        ]
    results: dict[str, R] = {}
    for key, future in futures:
        results[key] = future.result()
    return [results[task.stable_key] for task in ordered]

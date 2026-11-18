"""线程池并发执行器：稳定排序键保证并发/串行逐字段一致。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Sequence, TypeVar

from .fingerprint import stable_order_key

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Task(Generic[T]):
    """携带稳定排序键的候选任务。"""
    sample_id: str
    start: int
    stage: str
    payload: T

    @property
    def order_key(self) -> str:
        return stable_order_key(self.sample_id, self.start, self.stage)


def run_tasks(tasks: Sequence[Task[T]], worker: Callable[[Task[T]], R],
              max_workers: int) -> List[R]:
    """并发执行并按稳定排序键回收结果。

    线程调度顺序不影响最终顺序：所有 future 完成后按
    (样本 ID, 起始位置, 阶段名) 统一排序，与单线程执行完全一致。
    """
    ordered = sorted(tasks, key=lambda task: task.order_key)
    if max_workers <= 1 or len(ordered) <= 1:
        return [worker(task) for task in ordered]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [(task, pool.submit(worker, task)) for task in ordered]
    results = [(task.order_key, future.result()) for task, future in futures]
    results.sort(key=lambda item: item[0])
    return [result for _, result in results]

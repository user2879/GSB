"""有界线程池执行器：并发度可审计，任务带稳定排序键，结果顺序与线程调度无关。"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Task:
    sort_key: tuple
    fn: Callable[[], Any]
    label: str

    def __call__(self) -> Any:
        return self.fn()


def stable_task_key(sample: str, stage: str,
                    tile: tuple[int, int] | None = None) -> tuple:
    """由样本名、分块行列号与阶段名派生的稳定排序键。"""

    if tile is None:
        return (sample, -1, -1, stage)
    return (sample, tile[0], tile[1], stage)


class BoundedPool:
    """线程数硬上限取自配置；记录观测到的最大实际并发数用于并发报告。"""

    def __init__(self, max_workers: int, work_ms_per_task: int = 0):
        if max_workers <= 0:
            raise ValueError("max_workers 必须为正整数")
        self.max_workers = max_workers
        self.work_ms_per_task = work_ms_per_task
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()

    def map_ordered(self, tasks: Iterable[Task]) -> list[tuple[tuple, Any]]:
        ordered = sorted(tasks, key=lambda task: task.sort_key)
        results: list[tuple[tuple, Any] | None] = [None] * len(ordered)
        if not ordered:
            return []

        def wrapped(index: int, task: Task) -> tuple[int, Any]:
            with self._lock:
                self._active += 1
                self._peak = max(self._peak, self._active)
            try:
                if self.work_ms_per_task:
                    time.sleep(self.work_ms_per_task / 1000.0)
                return index, task()
            finally:
                with self._lock:
                    self._active -= 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(wrapped, index, task)
                       for index, task in enumerate(ordered)]
            for future in concurrent.futures.as_completed(futures):
                index, value = future.result()
                results[index] = (ordered[index].sort_key, value)
        return [item for item in results if item is not None]  # type: ignore[list-item]

    @property
    def peak_concurrency(self) -> int:
        return self._peak

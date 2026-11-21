"""并发执行器：任务携带稳定排序键，回收顺序与单线程稳定顺序严格一致。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_keyed(tasks: Iterable[tuple[str, T]], func: Callable[[T], R],
              concurrency: int) -> list[tuple[str, R]]:
    """tasks 已按稳定顺序给出；并发执行后按排序键升序回收。"""
    ordered = list(tasks)
    if concurrency <= 1 or len(ordered) <= 1:
        return [(key, func(payload)) for key, payload in ordered]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(func, payload): key for key, payload in ordered}
        collected: list[tuple[str, R]] = []
        for future, key in futures.items():
            collected.append((key, future.result()))
    collected.sort(key=lambda item: item[0])
    return collected

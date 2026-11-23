"""线程池并发执行器。

每个任务携带由 pair_id 与阶段名派生的稳定排序键；
无论并发度或调度顺序如何，最终结果都按排序键回收，保证逐字段一致。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


def stable_sort_key(pair_id: str, stage: str) -> str:
    return f"{stage}::{pair_id}"


def run_tasks(tasks: list[tuple[str, Callable[[], T]]], max_workers: int,
              parallel: bool) -> list[tuple[str, T]]:
    """tasks: [(sort_key, callable), ...]；返回按 sort_key 排序的结果。

    parallel=False 或 max_workers<=1 时退化为串行，输出顺序完全相同。
    单任务异常被捕获并以 (sort_key, exc) 形式回传，由上层映射业务码，
    保证线程异常不会破坏整体顺序或中断整批。
    """
    ordered = sorted(tasks, key=lambda t: t[0])
    results: list[tuple[str, T | Exception]] = []

    def _guard(fn: Callable[[], T]) -> Callable[[], T | Exception]:
        def _wrapped() -> T | Exception:
            try:
                return fn()
            except Exception as exc:  # 真实输入故障在此隔离
                return exc
        return _wrapped

    if parallel and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_guard(fn)) for _, fn in ordered]
            for (key, _), fut in zip(ordered, futures):
                results.append((key, fut.result()))
    else:
        for key, fn in ordered:
            results.append((key, _guard(fn)()))

    results.sort(key=lambda r: r[0])
    return results  # type: ignore[return-value]

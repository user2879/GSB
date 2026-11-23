"""线程池并发特征提取，结果按稳定排序键重排，保证与单线程逐字段一致。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class TaskEnvelope:
    sort_key: str
    round: int
    func: object
    kwargs: dict


def run_tasks(tasks: list[TaskEnvelope], max_workers: int) -> list:
    # 提交前按排序键稳定排序；收集后再次按同一排序键重排，
    # 无论线程实际完成顺序如何，输出顺序恒定。
    ordered = sorted(tasks, key=lambda t: t.sort_key)
    if max_workers <= 1:
        return [task.func(**task.kwargs) for task in ordered]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(task.func, **task.kwargs) for task in ordered]
        results = [future.result() for future in futures]
    results.sort(key=lambda item: getattr(item, "sort_key", ""))
    return results


def effective_workers(configured: int, n_tasks: int) -> int:
    # 最大并发数取自配置且不得超过。
    return max(1, min(configured, n_tasks))

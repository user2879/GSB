"""首响预算语义与受控线程池。

任务携带由段标识与阶段名派生的稳定排序键，线程池最大并发严格取自
配置；预算耗尽时保留待办并返回 E_LATENCY_FALLBACK 的先前可用摘要。
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .errors import E_LATENCY_FALLBACK
from .utils import stable_key


class BudgetTask:
    def __init__(self, audio_id, segment_index, stage, func, kwargs=None):
        self.audio_id = audio_id
        self.segment_index = int(segment_index)
        self.stage = stage
        self.sort_key = stable_key(audio_id, segment_index, stage)
        self.func = func
        self.kwargs = dict(kwargs or {})


class BudgetExecutor:
    def __init__(self, max_workers, budget_ms):
        self.max_workers = max(1, int(max_workers))
        self.budget_s = float(budget_ms) / 1000.0
        self.observed_max_workers = 0
        self.active = 0
        self._lock_peak = 0

    def run(self, tasks, previous_summary=None):
        """按稳定键排序后提交，返回 (结果表, 待办任务, 降级事件)。"""
        ordered = sorted(tasks, key=lambda task: task.sort_key)
        results = {}
        pending = []
        fallback = None
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for task in ordered:
                if time.perf_counter() - start >= self.budget_s and not futures:
                    pending.extend(ordered[len(futures):])
                    break
                future = pool.submit(task.func, **task.kwargs)
                futures[future] = task
            for future in as_completed(futures):
                task = futures[future]
                if time.perf_counter() - start >= self.budget_s and not future.done():
                    pending.append(task)
                    future.cancel()
                    continue
                try:
                    value = future.result()
                except Exception as exc:
                    value = ("__error__", exc)
                results[task.sort_key] = {
                    "sort_key": task.sort_key,
                    "stage": task.stage,
                    "value": value,
                }
        submitted = len(ordered)
        if pending:
            fallback = {
                "code": E_LATENCY_FALLBACK,
                "message": "首响预算内无法完成全部任务，返回先前可用结果并保留待办",
                "budget_ms": self.budget_s * 1000.0,
                "completed": len(results),
                "submitted": submitted,
                "pending_sort_keys": [task.sort_key for task in pending],
                "previous_summary": previous_summary,
            }
        return results, pending, fallback

    def run_sequential_under_budget(self, tasks, previous_summary=None):
        """确定性单线程模式：严格按稳定键执行，预算耗尽即停。"""
        ordered = sorted(tasks, key=lambda task: task.sort_key)
        results = {}
        pending = []
        start = time.perf_counter()
        for task in ordered:
            if time.perf_counter() - start >= self.budget_s:
                pending.append(task)
                continue
            try:
                value = task.func(**task.kwargs)
            except Exception as exc:
                value = ("__error__", exc)
            results[task.sort_key] = {
                "sort_key": task.sort_key,
                "stage": task.stage,
                "value": value,
            }
        fallback = None
        if pending:
            fallback = {
                "code": E_LATENCY_FALLBACK,
                "message": "首响预算内无法完成全部任务，返回先前可用结果并保留待办",
                "budget_ms": self.budget_s * 1000.0,
                "completed": len(results),
                "submitted": len(ordered),
                "pending_sort_keys": [task.sort_key for task in pending],
                "previous_summary": previous_summary,
            }
        return results, pending, fallback

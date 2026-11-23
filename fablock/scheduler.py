"""单批打分点数预算 + 单查询块配额的并行调度器。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import Code


@dataclass(frozen=True)
class ScoreTask:
    query_block: int
    key_block: int
    points: int


@dataclass
class ScheduleEvent:
    code: str
    query_block: int
    message: str

    def to_dict(self):
        return {"code": self.code, "query_block": self.query_block,
                "message": self.message}


class ConcurrencyTracker:
    """记录线程池实际并发峰值，用于校验不超过配置上限。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def __enter__(self):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self._active -= 1
        return False


class BudgetScheduler:
    """按查询块序号稳定顺序分配配额，维护全局剩余点数预算与待办队列。"""

    def __init__(self, cfg, num_query_blocks: int, num_key_blocks: int):
        self.workers = cfg.thread_pool_workers
        self.budget = cfg.point_budget_per_batch
        self.remaining = cfg.point_budget_per_batch
        self.num_key_blocks = num_key_blocks
        self.points_per_pair = cfg.query_block_len * cfg.key_block_len
        # 单查询块配额：预算在查询块间稳定均分（向上取整）
        self.quota = -(-cfg.point_budget_per_batch // max(1, num_query_blocks))
        self.pending: list = []
        self.events: list = []
        self.tracker = ConcurrencyTracker()

    def plan(self, needed_pairs) -> list:
        """needed_pairs: 本批需要计算的 (query_block, key_block) 列表。
        返回可执行任务列表；超配额块记 E_NODE_BUDGET_EXHAUSTED，
        预算耗尽的剩余任务进入待办并记 E_POINT_BUDGET_PENDING。"""
        runnable = []
        # 待办优先
        carry = [(t.query_block, t.key_block) for t in self.pending]
        self.pending = []
        per_block = {}
        for qi, ki in carry + sorted(needed_pairs):
            per_block.setdefault(qi, []).append(ki)
        for qi in sorted(per_block):
            kis = per_block[qi]
            block_points = len(kis) * self.points_per_pair
            if block_points > self.quota:
                self.events.append(ScheduleEvent(
                    Code.E_NODE_BUDGET_EXHAUSTED, qi,
                    f"查询块 {qi} 需要 {block_points} 点，超过单块配额 {self.quota}"))
                continue
            admitted = []
            for ki in kis:
                if self.remaining >= self.points_per_pair:
                    admitted.append(ki)
                    self.remaining -= self.points_per_pair
                else:
                    self.pending.append(ScoreTask(qi, ki, self.points_per_pair))
            if len(admitted) < len(kis) and not any(
                    e.code == Code.E_POINT_BUDGET_PENDING for e in self.events):
                self.events.append(ScheduleEvent(
                    Code.E_POINT_BUDGET_PENDING, qi,
                    f"全局预算耗尽，{len(kis) - len(admitted)} 个任务转入待办队列"))
            runnable.extend(ScoreTask(qi, ki, self.points_per_pair)
                            for ki in admitted)
        return runnable

    def execute(self, tasks, fn) -> dict:
        """线程池并发执行打分任务；结果按 (query_block, key_block) 稳定归集。

        fn(task) 的异常被捕获并放入 results 的异常表，由上层做故障隔离。
        """
        results = {}
        errors = {}

        def run(task):
            with self.tracker:
                return fn(task)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            future_map = {pool.submit(run, t): t for t in tasks}
            for fut, task in future_map.items():
                try:
                    results[(task.query_block, task.key_block)] = fut.result()
                except Exception as exc:  # noqa: BLE001 - 故障隔离在上层
                    errors[(task.query_block, task.key_block)] = exc
        return {"results": results, "errors": errors,
                "peak_concurrency": self.tracker.peak}

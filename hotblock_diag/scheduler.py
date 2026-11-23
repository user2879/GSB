"""Budget-constrained parallel scheduler for block-pair scoring tasks.

The scheduler submits work to a ThreadPoolExecutor, but budget accounting is
performed by the orchestrator thread in stable-key order before any task is
submitted. Thread completion order therefore cannot change the deduction
sequence, which is identical to a serial walk over the stable keys.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import (E_QBLOCK_BUDGET_EXHAUSTED, E_SCORE_BUDGET_EXHAUSTED,
                     FailureList)


@dataclass
class ScoreTask:
    qb: int
    kb: int
    payload: dict[str, Any]

    @property
    def key(self) -> tuple[int, int]:
        return (self.qb, self.kb)


@dataclass
class BatchResult:
    results: dict[tuple[int, int], Any] = field(default_factory=dict)
    pending: list[ScoreTask] = field(default_factory=list)
    deductions: list[dict[str, int]] = field(default_factory=list)
    exhausted_q: set[int] = field(default_factory=set)
    global_exhausted: bool = False


class Scheduler:
    def __init__(self, threads: int, batch_budget: int, qblock_quota: int,
                 failures: FailureList, stage: str = "block_scoring"):
        self.threads = max(1, threads)
        self.batch_budget = batch_budget
        self.qblock_quota = qblock_quota
        self.failures = failures
        self.stage = stage
        self.total_consumed = 0
        self.total_deductions: list[dict[str, int]] = []
        self.batches_executed = 0

    def run_batch(self, tasks: list[ScoreTask],
                  worker: Callable[[ScoreTask], Any]) -> BatchResult:
        self.batches_executed += 1
        ordered = sorted(tasks, key=lambda task: task.key)
        result = BatchResult()
        remaining_global = self.batch_budget
        per_q_remaining: dict[int, int] = {}
        accepted: list[ScoreTask] = []
        q_failure_seen: set[int] = set()
        global_failure_seen = False

        for task in ordered:
            qb, kb = task.key
            quota = per_q_remaining.get(qb, self.qblock_quota)
            if quota <= 0 or qb in result.exhausted_q:
                result.exhausted_q.add(qb)
                result.pending.append(task)
                if qb not in q_failure_seen:
                    q_failure_seen.add(qb)
                    self.failures.add(
                        E_QBLOCK_BUDGET_EXHAUSTED, self.stage, f"qb={qb}",
                        detail=f"qblock quota {self.qblock_quota} exhausted; kb={kb} deferred",
                    )
                continue
            if remaining_global <= 0:
                result.global_exhausted = True
                result.pending.append(task)
                if not global_failure_seen:
                    global_failure_seen = True
                    self.failures.add(
                        E_SCORE_BUDGET_EXHAUSTED, self.stage, f"qb={qb};kb={kb}",
                        detail=f"global batch budget {self.batch_budget} exhausted",
                    )
                continue
            per_q_remaining[qb] = quota - 1
            remaining_global -= 1
            deduction = {"seq": len(self.total_deductions), "qb": qb, "kb": kb,
                         "global_remaining_after": remaining_global,
                         "qblock_remaining_after": quota - 1}
            result.deductions.append(deduction)
            self.total_deductions.append(dict(deduction))
            self.total_consumed += 1
            accepted.append(task)

        if accepted:
            with ThreadPoolExecutor(max_workers=self.threads) as pool:
                futures = {pool.submit(worker, task): task for task in accepted}
                for future in futures:
                    task = futures[future]
                    result.results[task.key] = future.result()
        return result

"""受预算约束的增量重算调度。

- 全局并发上限：同一波次内通过有界线程池并发执行；
- 每会话重算预算：按重算条目数计，预算耗尽时剩余任务进入排队，
  记录等待开始时刻，绝不绕过预算继续重算；
- 任务可声明依赖（seq），依赖不同波次的任务按波次串行推进；
- 每个重算任务输出开始/结束单调时钟毫秒、所属条目或视图及所用预算。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import BudgetError


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


@dataclass
class Task:
    seq: int
    kind: str  # "index" | "view"
    target: str  # 例如 "by_label:Person" 或视图名
    fn: Callable[[], Any]
    deps: tuple[int, ...] = field(default_factory=tuple)


class BudgetScheduler:
    def __init__(self, concurrency: int = 4, budget: int = 100):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self.budget = budget

    def execute(self, tasks: list[Task]) -> dict[str, Any]:
        if self.budget <= 0 and tasks:
            raise BudgetError(
                "recompute budget is zero or negative",
                [
                    {"element": t.target, "field": "budget", "message": "no recompute budget left"}
                    for t in tasks
                ],
            )
        runnable = list(tasks[: self.budget])
        queued = list(tasks[self.budget :])

        ledger: list[dict[str, Any]] = []
        queued_at = monotonic_ms()
        for task in queued:
            ledger.append(
                {
                    "seq": task.seq,
                    "kind": task.kind,
                    "target": task.target,
                    "status": "queued",
                    "queued_at_ms": queued_at,
                    "budget": 0,
                }
            )

        done = self._run_waves(runnable)
        ledger = done + ledger
        return {
            "ledger": ledger,
            "recomputed": len(runnable),
            "skipped": len(queued),
        }

    def _run_waves(self, tasks: list[Task]) -> list[dict[str, Any]]:
        """按依赖分层（波次）执行；波次内受并发上限约束，波次间保持依赖顺序。"""
        levels: dict[int, int] = {}
        by_seq = {t.seq: t for t in tasks}
        for task in tasks:
            deps_in = [d for d in task.deps if d in by_seq]
            levels[task.seq] = 1 + max((levels[d] for d in deps_in), default=-1)
        waves: dict[int, list[Task]] = {}
        for task in tasks:
            waves.setdefault(levels[task.seq], []).append(task)

        results: dict[int, dict[str, Any]] = {}

        def run_one(task: Task) -> dict[str, Any]:
            start = monotonic_ms()
            task.fn()
            end = monotonic_ms()
            return {
                "seq": task.seq,
                "kind": task.kind,
                "target": task.target,
                "status": "done",
                "start_ms": start,
                "end_ms": end,
                "budget": 1,
            }

        for level in sorted(waves):
            wave = sorted(waves[level], key=lambda t: t.seq)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                for entry in pool.map(run_one, wave):
                    results[entry["seq"]] = entry
        return [results[t.seq] for t in tasks]

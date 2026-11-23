"""单批片段处理预算与待办保留调度器。

按轮次稳定顺序在失效轮次间分配预算；预算不足时保留待处理队列，
下批优先处理，并以 E_CALIB_BUDGET_PENDING 标记未完成项。
绝不允许跳过失效轮次换取预算收敛。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatchLedger:
    batch_no: int
    budget: int
    consumed: int = 0
    reused: int = 0
    pending_rounds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch_no": self.batch_no,
            "budget": self.budget,
            "consumed": self.consumed,
            "remaining": self.budget - self.consumed,
            "reused_hits": self.reused,
            "pending_rounds": self.pending_rounds,
        }


@dataclass
class Scheduler:
    batch_budget: int
    batch_no: int = 0
    history: list[BatchLedger] = field(default_factory=list)
    current: BatchLedger | None = None

    def start_batch(self) -> BatchLedger:
        self.batch_no += 1
        self.current = BatchLedger(batch_no=self.batch_no, budget=self.batch_budget)
        return self.current

    def finish_batch(self) -> BatchLedger:
        assert self.current is not None
        self.history.append(self.current)
        return self.current

    def take(self, rnd: int, is_reuse: bool) -> str:
        """返回 'reuse' / 'compute' / 'pending'。"""
        assert self.current is not None
        if is_reuse:
            self.current.reused += 1
            return "reuse"
        if self.current.consumed >= self.batch_budget:
            if rnd not in self.current.pending_rounds:
                self.current.pending_rounds.append(rnd)
            return "pending"
        self.current.consumed += 1
        return "compute"

    def summarize(self) -> list[dict]:
        return [batch.to_dict() for batch in self.history]

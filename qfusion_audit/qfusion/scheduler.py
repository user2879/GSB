"""单批重算点数预算与待办保留调度器（要求 6/7 并发部分）。

工作单元为 (request_id, item_id)，稳定顺序为请求 ID 字典序、物品 ID 字典序。
每单元消耗 1 个重算点；预算不足的单元保留在待办队列并以
E_RECOMPUTE_BUDGET_PENDING 标记，下一批优先处理待办。
并发度取自配置且绝不超过；任务携带由请求 ID、物品 ID、阶段名派生的稳定键，
结果按稳定键回收，故并发与串行逐字段一致。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .errors import E_RECOMPUTE_BUDGET_PENDING, Failure
from .fixedpoint import STAGE_ORDER, stable_key


@dataclass
class WorkUnit:
    request_id: str
    item_id: str

    @property
    def key(self) -> str:
        # 阶段名纳入稳定排序键（要求 7）：取四阶段拼串保持单键派生。
        return stable_key(self.request_id, self.item_id,
                          "|".join(STAGE_ORDER))


@dataclass
class BatchLedger:
    batch_id: int
    budget: int
    consumed_points: int
    processed: List[str]
    pending: List[Dict[str, str]]
    pending_failures: List[Dict[str, Any]]
    sources: Dict[str, str]
    concurrency_used: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "budget": self.budget,
            "consumed_points": self.consumed_points,
            "processed": self.processed,
            "pending": self.pending,
            "pending_failures": self.pending_failures,
            "sources": self.sources,
            "concurrency_used": self.concurrency_used,
        }


class RecomputeScheduler:
    def __init__(self, units: List[WorkUnit], concurrency: int,
                 worker: Callable[[WorkUnit], Any]):
        # 稳定顺序：请求 ID、物品 ID 字典序。
        self.todo: List[WorkUnit] = sorted(units, key=lambda u: u.key)
        self.concurrency = concurrency
        self.worker = worker
        self.results: Dict[str, Any] = {}
        self.batch_ledgers: List[BatchLedger] = []
        self._sources: Dict[str, str] = {}

    def run_batch(self, budget: int) -> BatchLedger:
        batch_id = len(self.batch_ledgers) + 1
        processed: List[str] = []
        pending: List[Dict[str, str]] = []
        pending_failures: List[Dict[str, Any]] = []
        sources: Dict[str, str] = {}

        chosen: List[WorkUnit] = []
        while self.todo and len(chosen) < budget:
            chosen.append(self.todo.pop(0))
        # 预算不足：剩余进入待办并标记。
        for unit in list(self.todo):
            pending.append({"request_id": unit.request_id,
                            "item_id": unit.item_id})
            pending_failures.append(Failure(
                E_RECOMPUTE_BUDGET_PENDING, unit.key,
                "本批重算点数预算不足，保留至下一批优先处理"
            ).to_dict())

        concurrency_used = min(self.concurrency, max(1, len(chosen)))
        if chosen:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                future_map = {pool.submit(self.worker, u): u for u in chosen}
                buffered: List[Tuple[str, Any]] = []
                for fut in future_map:
                    unit = future_map[fut]
                    buffered.append((unit.key, fut.result()))
            # 按稳定键回收，消除线程完成顺序的不确定性。
            for key, result in sorted(buffered, key=lambda kv: kv[0]):
                self.results[key] = result

        for unit in chosen:
            processed.append(unit.key)
            sources[unit.key] = self._sources.get(unit.key, "fresh")
            self._sources[unit.key] = "recomputed"

        ledger = BatchLedger(
            batch_id=batch_id, budget=budget,
            consumed_points=len(processed), processed=processed,
            pending=pending, pending_failures=pending_failures,
            sources=sources, concurrency_used=concurrency_used)
        self.batch_ledgers.append(ledger)
        return ledger

    @property
    def drained(self) -> bool:
        return not self.todo

    def run_to_drain(self, batch_budget: int) -> List[BatchLedger]:
        ledgers = []
        while not self.drained:
            ledgers.append(self.run_batch(batch_budget))
        return ledgers

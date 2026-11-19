"""并发调度与资源预算。

- ``thread_budget``：工作线程数；1 -> 串行；>1 -> ThreadPoolExecutor。
- ``memory_budget``：允许同时驻留的路径缓存条数；超额按 LRU 淘汰。
- 预算不足按固定优先级降级：先淘汰缓存 -> 再退化为串行 -> 完全拒绝。
- 所有结果按查询键排序归并，并发只改变执行顺序，不改变结果内容。
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from .errors import BudgetDeniedError

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class DegradationRecord:
    reason: str
    action: str
    detail: Dict[str, Any] = field(default_factory=dict)
    expanded_before: int = 0
    expanded_after: int = 0

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    thread_budget: int = 4
    memory_budget: int = 64
    max_expansion_budget: Optional[int] = None
    serial_fallback: bool = True
    allow_cache_eviction: bool = True

    def validate(self) -> None:
        if self.thread_budget < 1 or self.memory_budget < 1:
            raise BudgetDeniedError(
                "资源预算非法（线程数与内存预算必须 >=1）",
                thread_budget=self.thread_budget,
                memory_budget=self.memory_budget,
            )


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: "OrderedDict[K, V]" = OrderedDict()
        self.evictions: List[K] = []
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> Optional[V]:
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: K, value: V) -> List[K]:
        evicted: List[K] = []
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            old_key, _value = self._store.popitem(last=False)
            self.evictions.append(old_key)
            evicted.append(old_key)
        return evicted

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> List[K]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()


@dataclass
class Scheduler:
    budget: Budget
    degradation_log: List[DegradationRecord] = field(default_factory=list)
    total_expanded: int = 0
    last_expanded: int = 0

    def __post_init__(self) -> None:
        self.budget.validate()

    def run_queries(
        self,
        items: List[Tuple[str, Callable[[], Any]]],
        expansion_supplier: Optional[Callable[[Any], int]] = None,
    ) -> Dict[str, Any]:
        ordered = sorted(items, key=lambda item: item[0])
        effective_workers = self.budget.thread_budget
        if len(ordered) <= 1 or self.budget.thread_budget == 1:
            effective_workers = 1
        elif len(ordered) < self.budget.thread_budget:
            effective_workers = len(ordered)
            self._degrade(
                reason="thread_oversubscribed",
                action="resize_worker_pool",
                detail={"requested": self.budget.thread_budget, "effective": effective_workers},
            )

        before = self.total_expanded
        if effective_workers == 1:
            if self.budget.thread_budget > 1:
                self._degrade(
                    reason="thread_budget_unavailable",
                    action="serial_fallback",
                    detail={"requested": self.budget.thread_budget},
                )
            results = {key: worker() for key, worker in ordered}
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {pool.submit(worker): key for key, worker in ordered}
                results = {}
                for future in futures:
                    key = futures[future]
                    results[key] = future.result()
        after = self.total_expanded
        self.last_expanded = after - before
        return {key: results[key] for key in sorted(results)}

    def run_regions(
        self,
        region_items: List[Tuple[int, Callable[[], Any]]],
    ) -> Dict[int, Any]:
        ordered = sorted(region_items, key=lambda item: item[0])
        workers = min(self.budget.thread_budget, len(ordered)) if ordered else 1
        if workers < self.budget.thread_budget:
            self._degrade(
                reason="region_worker_resize",
                action="resize_worker_pool",
                detail={"requested": self.budget.thread_budget, "effective": max(workers, 1)},
            )
        if workers <= 1:
            return {region_id: worker() for region_id, worker in ordered}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker): region_id for region_id, worker in ordered}
            merged = {futures[future]: future.result() for future in futures}
        return {region_id: merged[region_id] for region_id in sorted(merged)}

    def account_expansion(self, expanded: int) -> None:
        self.total_expanded += expanded
        if (
            self.budget.max_expansion_budget is not None
            and self.total_expanded > self.budget.max_expansion_budget
        ):
            raise BudgetDeniedError(
                "展开节点预算被拒绝",
                total_expanded=self.total_expanded,
                expansion_budget=self.budget.max_expansion_budget,
            )

    def enforce_memory(self, cache: LRUCache[Any, Any]) -> List[Any]:
        if len(cache) <= self.budget.memory_budget:
            return []
        if not self.budget.allow_cache_eviction:
            raise BudgetDeniedError(
                "内存预算不足且禁止缓存淘汰",
                cache_size=len(cache),
                memory_budget=self.budget.memory_budget,
            )
        before = len(cache)
        evicted: List[Any] = []
        while len(cache) > self.budget.memory_budget:
            key, _value = cache._store.popitem(last=False)
            cache.evictions.append(key)
            evicted.append(key)
        self._degrade(
            reason="memory_budget_exceeded",
            action="lru_eviction",
            detail={"before": before, "after": len(cache), "evicted": evicted},
        )
        return evicted

    def _degrade(self, reason: str, action: str, detail: Dict[str, Any]) -> None:
        self.degradation_log.append(
            DegradationRecord(
                reason=reason,
                action=action,
                detail=detail,
                expanded_before=self.total_expanded,
                expanded_after=self.total_expanded,
            )
        )

    def report(self) -> Dict[str, Any]:
        return {
            "budget": asdict(self.budget),
            "total_expanded": self.total_expanded,
            "last_expanded": self.last_expanded,
            "degradation": [record.to_payload() for record in self.degradation_log],
        }

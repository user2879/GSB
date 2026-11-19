"""共享中间缓冲的确定性驻留、LRU 回收与驻留账本。

驻留决策不依赖线程完成顺序：调度前先以任务键的时间轴顺序做一次规范模拟
（canonical plan），运行时缓冲池严格按该计划执行 HIT/STORE/EVICT/RECOMPUTE。
因此线程数变化只改变执行并发，不改变任何驻留/复用结论，两次运行账本逐字节
一致。被淘汰后再次访问一律记为 RECOMPUTE（重新计算），绝不可能误记为复用。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import E_BUDGET_DENIED, ErrorRecord

BYTES_PER_FLOAT64 = 8
RAW_GRID = 480  # 10ms 网格，保证相邻任务的上下文切块共享同一批键
RAW_KEY_PREFIX = "raw"
RESAMPLE_KEY_PREFIX = "rsm"  # 48k 直通：重采样为恒等映射，仍独立记账
INTEGRATION_KEY_PREFIX = "intg"


def _grid_floor(x: int, grid: int = RAW_GRID) -> int:
    return (x // grid) * grid


def raw_chunk_keys(context_start: int, context_end: int) -> List[Tuple[str, int, int]]:
    """把上下文按固定网格切成可复用缓冲键 (key, chunk_start, chunk_end)。"""
    keys: List[Tuple[str, int, int]] = []
    pos = _grid_floor(context_start)
    while pos < context_end:
        nxt = pos + RAW_GRID  # 始终按完整网格块取数，键在任务间稳定可共享
        keys.append((f"{RAW_KEY_PREFIX}:{pos:08d}:{nxt:08d}", pos, nxt))
        pos = nxt
    return keys


def _raw_byte_size(channels: int, start: int, end: int) -> int:
    return channels * (end - start) * BYTES_PER_FLOAT64


@dataclass
class TaskCachePlan:
    task_key: str
    denied: bool = False
    denial_reason: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    recomputes: int = 0
    uncached: int = 0
    evictions: int = 0
    required_bytes: int = 0


@dataclass
class CachePlan:
    budget_bytes: Optional[int]
    ledger: List[Dict[str, Any]] = field(default_factory=list)
    task_plans: Dict[str, TaskCachePlan] = field(default_factory=dict)
    denied_records: List[ErrorRecord] = field(default_factory=list)
    final_resident: List[str] = field(default_factory=list)
    total_stores: int = 0
    total_hits: int = 0
    total_recomputes: int = 0
    total_uncached: int = 0
    total_evictions: int = 0


def _evict_event(task_key: str, key: str, bytes_: int, reason: str) -> Dict[str, Any]:
    return {
        "task_key": task_key,
        "event": "EVICT",
        "buffer_key": key,
        "bytes": bytes_,
        "reason": reason,
    }


def build_cache_plan(tasks: List, channels: int, budget_bytes: Optional[int]) -> CachePlan:
    """以时间轴顺序规范模拟 LRU，生成确定性驻留账本与每任务计划。"""
    plan = CachePlan(budget_bytes=budget_bytes)
    resident: Dict[str, int] = {}  # key -> bytes；dict 顺序即 LRU（旧->新）

    for task in tasks:
        tp = TaskCachePlan(task_key=task.task_key)

        # 该任务需要的全部缓冲（键顺序固定，保证账本确定性）
        needs: List[Tuple[str, int]] = []
        for key, cstart, cend in raw_chunk_keys(task.context_start, task.context_end):
            raw_bytes = _raw_byte_size(channels, cstart, cend)
            needs.append((key, raw_bytes))
            needs.append((key.replace(RAW_KEY_PREFIX, RESAMPLE_KEY_PREFIX, 1), raw_bytes))
        intg_bytes = max(task.buffer_declaration["integration_bytes"], BYTES_PER_FLOAT64)
        needs.append((f"{INTEGRATION_KEY_PREFIX}:{task.task_key}", intg_bytes))
        tp.required_bytes = sum(b for _, b in needs)

        for key, size in needs:
            if budget_bytes is not None and size > budget_bytes:
                # 单缓冲超出总预算：记录 E_BUDGET_DENIED，走无缓存重算降级路径
                tp.denied = True
                tp.denial_reason = "single_buffer_exceeds_budget"
                plan.denied_records.append(
                    ErrorRecord(
                        code=E_BUDGET_DENIED,
                        message="单个中间缓冲超过内存总预算，拒绝驻留",
                        location={"task_key": task.task_key, "buffer_key": key},
                        evidence={"buffer_bytes": size, "budget_bytes": budget_bytes},
                        recovery="degrade_to_uncached_recompute",
                    )
                )
                tp.uncached += 1
                plan.total_uncached += 1
                tp.events.append(
                    {
                        "task_key": task.task_key,
                        "event": "DENY",
                        "buffer_key": key,
                        "bytes": size,
                        "reason": "exceeds_total_budget",
                    }
                )
                continue

            if key in resident:
                resident.pop(key)
                resident[key] = size  # 提到最新
                tp.hits += 1
                plan.total_hits += 1
                tp.events.append(
                    {"task_key": task.task_key, "event": "HIT", "buffer_key": key, "bytes": size}
                )
                continue

            # 未驻留：可能是首次 MISS，也可能是曾被淘汰后的 RECOMPUTE
            ever_seen = any(
                e["buffer_key"] == key and e["event"] in ("STORE", "RECOMPUTE")
                for e in plan.ledger
            )
            if budget_bytes is not None:
                while sum(resident.values()) + size > budget_bytes and resident:
                    old_key, old_size = next(iter(resident.items()))
                    resident.pop(old_key)
                    tp.evictions += 1
                    plan.total_evictions += 1
                    tp.events.append(
                        _evict_event(task.task_key, old_key, old_size, "lru_budget_pressure")
                    )
            if budget_bytes is not None and sum(resident.values()) + size > budget_bytes:
                tp.denied = True
                tp.denial_reason = "unable_to_reserve_after_lru"
                plan.denied_records.append(
                    ErrorRecord(
                        code=E_BUDGET_DENIED,
                        message="LRU 回收后仍无法满足驻留预算，拒绝驻留并重算",
                        location={"task_key": task.task_key, "buffer_key": key},
                        evidence={"buffer_bytes": size, "budget_bytes": budget_bytes},
                        recovery="degrade_to_uncached_recompute",
                    )
                )
                tp.uncached += 1
                plan.total_uncached += 1
                tp.events.append(
                    {
                        "task_key": task.task_key,
                        "event": "DENY",
                        "buffer_key": key,
                        "bytes": size,
                        "reason": "reserve_failed_after_lru",
                    }
                )
                continue

            resident[key] = size
            plan.total_stores += 1
            if ever_seen:
                tp.recomputes += 1
                plan.total_recomputes += 1
                tp.events.append(
                    {
                        "task_key": task.task_key,
                        "event": "RECOMPUTE",
                        "buffer_key": key,
                        "bytes": size,
                        "reason": "previously_evicted",
                    }
                )
            else:
                tp.misses += 1
                tp.events.append(
                    {"task_key": task.task_key, "event": "STORE", "buffer_key": key, "bytes": size}
                )

        plan.task_plans[task.task_key] = tp
        plan.ledger.extend(tp.events)

    plan.final_resident = list(resident.keys())
    return plan


class BufferPool:
    """运行时缓冲池：只执行规范计划，不自行决定驻留，保证并发下确定性。"""

    def __init__(self, plan: CachePlan):
        self.plan = plan
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def acquire(self, task_key: str, key: str, loader) -> Tuple[Any, str]:
        """按计划获取缓冲。返回 (buffer, event_name)。loader 仅在非 HIT 时调用。"""
        with self._lock:
            if key in self._store:
                return self._store[key], "HIT"
        value = loader()
        # 是否驻留由规范计划中该键的事件决定；DENY 时仍返回重算结果，只是不缓存
        planned = [e for e in self.plan.task_plans[task_key].events if e["buffer_key"] == key]
        store_event = next((e for e in planned if e["event"] in ("STORE", "RECOMPUTE")), None)
        if store_event is not None:
            with self._lock:
                self._store[key] = value
            return value, store_event["event"]
        return value, "UNCACHED"

    def apply_evictions(self, task_key: str) -> List[str]:
        evicted: List[str] = []
        with self._lock:
            for e in self.plan.task_plans[task_key].events:
                if e["event"] == "EVICT" and e["buffer_key"] in self._store:
                    del self._store[e["buffer_key"]]
                    evicted.append(e["buffer_key"])
        return evicted

    def resident_keys(self) -> List[str]:
        return list(self._store.keys())

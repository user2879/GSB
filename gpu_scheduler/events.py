"""Immutable trace records produced by the deterministic engine."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

# Event kinds persisted in the ledger.
DISPATCH = "DISPATCH"
COMPLETE = "COMPLETE"
FAIL = "FAIL"
PREEMPT = "PREEMPT"
ENQUEUE = "ENQUEUE"
INVALIDATE = "INVALIDATE"
MARK_UNKNOWN = "MARK_UNKNOWN"


@dataclass(frozen=True)
class TraceEvent:
    round_no: int
    clock_ms: int
    kind: str
    task_id: str
    shard_index: int
    pool_id: Optional[str]
    gpu_count: int
    memory_mib: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PreemptionRecord:
    round_no: int
    clock_ms: int
    trigger_task_id: str
    trigger_shard_index: int
    victim_task_id: str
    victim_shard_index: int
    pool_id: str
    released_gpu: int
    released_memory_mib: int
    trigger_priority: int
    victim_priority: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PoolUsage:
    pool_id: str
    gpu_busy: int
    gpu_capacity: int
    memory_busy_mib: int
    memory_capacity_mib: int
    shares_busy: int
    shares_capacity: int
    concurrent_tasks: int
    concurrent_cap: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoundRecord:
    round_no: int
    clock_ms: int
    events: List[TraceEvent]
    preemption: Optional[PreemptionRecord]
    pool_usage: List[PoolUsage]
    running: List[str]
    waiting: List[str]
    global_gpu_busy: int
    global_shares_busy: int

    def to_dict(self) -> dict:
        return {
            "round_no": self.round_no,
            "clock_ms": self.clock_ms,
            "events": [event.to_dict() for event in self.events],
            "preemption": self.preemption.to_dict()
            if self.preemption else None,
            "pool_usage": [usage.to_dict() for usage in self.pool_usage],
            "running": list(self.running),
            "waiting": list(self.waiting),
            "global_gpu_busy": self.global_gpu_busy,
            "global_shares_busy": self.global_shares_busy,
        }

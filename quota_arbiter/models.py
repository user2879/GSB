"""租户、资源池、请求与结局的不可变数据模型。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tenant:
    tid: str
    quota: int
    concurrency_cap: int
    burst: int


@dataclass(frozen=True)
class Pool:
    pid: str
    capacity: int
    priority_order: tuple[str, ...]


@dataclass(frozen=True)
class Request:
    rid: str
    tid: str
    pid: str
    demand: int
    arrival: int
    priority: int


@dataclass
class Outcome:
    """单条请求的最终结局记录（一条请求至多一条）。"""

    rid: str
    tid: str
    pid: str
    arrival: int
    verdict: str
    reason: str | None
    basis: str
    decided_at: int
    granted_at: int | None
    completed_at: int | None
    wait_ms: int
    tenant_quota_remaining: int
    tenant_burst_remaining: int
    inflight: int
    pool_capacity_remaining: int
    demand: int
    enqueued: bool

    def canonical_tuple(self):
        return (
            self.rid,
            self.tid,
            self.pid,
            self.arrival,
            self.verdict,
            self.reason or "",
            self.basis,
            self.decided_at,
            -1 if self.granted_at is None else self.granted_at,
            -1 if self.completed_at is None else self.completed_at,
            self.wait_ms,
            self.tenant_quota_remaining,
            self.tenant_burst_remaining,
            self.inflight,
            self.pool_capacity_remaining,
            self.demand,
            1 if self.enqueued else 0,
        )


@dataclass
class QueueEvent:
    """排队轨迹：入队与唤醒各一条。"""

    rid: str
    tid: str
    pid: str
    kind: str  # ENQUEUE / WAKE / STAY
    at_ms: int
    wait_ms: int
    basis: str
    detail: str


@dataclass
class RunResult:
    outcomes: list[Outcome]
    queue_events: list[QueueEvent]
    final_tenant_quota: dict[str, int]
    final_tenant_burst: dict[str, int]
    final_pool_capacity: dict[str, int]
    final_inflight: dict[str, int]
    batches: list[dict]
    last_rid: str

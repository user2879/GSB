"""确定性仲裁状态机。

三条执行路径共用本引擎，仅喂入请求的批次边界不同，因此结局序列天然一致。
事件循环按规范序 (arrival, rid) 处理到达，按 (completion_time, rid) 处理完成；
唤醒选择使用资源池优先级序、请求优先级、到达时刻、请求标识的显式稳定排序，
不依赖任何 dict / set 的遍历顺序。
"""
from __future__ import annotations

from . import (
    REASON_BURST,
    REASON_POOL_CAPACITY,
    REASON_TENANT_QUOTA,
    VERDICT_GRANT,
    VERDICT_QUEUE,
    VERDICT_REJECT,
)
from .models import Outcome, QueueEvent, Request
from .util import hold_duration_ms


class ArbiterEngine:
    def __init__(self, tenants, pools):
        self.tenants = {t.tid: t for t in tenants}
        self.pools = {p.pid: p for p in pools}
        self.tenant_order = {t.tid: i for i, t in enumerate(sorted(tenants, key=lambda x: x.tid))}

        self.quota_remaining = {t.tid: t.quota for t in tenants}
        self.burst_remaining = {t.tid: t.burst for t in tenants}
        self.inflight = {t.tid: 0 for t in tenants}
        self.pool_remaining = {p.pid: p.capacity for p in pools}

        # 资源池 -> 仍在排队的请求标识
        self.queues: dict[str, list[str]] = {p.pid: [] for p in pools}
        self.requests_by_id: dict[str, Request] = {}
        self.outcomes: dict[str, Outcome] = {}
        self.queue_events: list[QueueEvent] = []
        # 活动授予 (completion_time, rid)
        self.active: list[tuple[int, str]] = []
        self._now = 0

    # ------------------------------------------------------------------ #
    # 稳定排序键
    # ------------------------------------------------------------------ #
    def _pool_priority_rank(self, pid: str, tid: str) -> int:
        order = self.pools[pid].priority_order
        if tid in order:
            return order.index(tid)
        # 未列入该池优先级序的租户排在末尾，再按全局租户序兜底
        return len(order) + self.tenant_order[tid]

    def _wake_key(self, req: Request):
        return (
            self._pool_priority_rank(req.pid, req.tid),
            req.priority,
            req.arrival,
            req.rid,
        )

    def _snapshot(self, req: Request):
        return {
            "tenant_quota_remaining": self.quota_remaining[req.tid],
            "tenant_burst_remaining": self.burst_remaining[req.tid],
            "inflight": self.inflight[req.tid],
            "pool_capacity_remaining": self.pool_remaining[req.pid],
        }

    # ------------------------------------------------------------------ #
    # 授予（仅做资源状态变更，返回 (成功, 使用突发, 完成时刻)）
    # ------------------------------------------------------------------ #
    def _apply_grant(self, req: Request, now: int):
        tenant = self.tenants[req.tid]
        pool = self.pools[req.pid]
        if self.inflight[req.tid] >= tenant.concurrency_cap:
            return False, False, -1
        if self.pool_remaining[req.pid] >= req.demand:
            self.pool_remaining[req.pid] -= req.demand
            self.inflight[req.tid] += 1
            complete_at = now + hold_duration_ms(req.demand, req.priority, req.rid)
            self.active.append((complete_at, req.rid))
            return True, False, complete_at
        if self.burst_remaining[req.tid] >= req.demand and req.demand <= pool.capacity:
            self.burst_remaining[req.tid] -= req.demand
            self.inflight[req.tid] += 1
            complete_at = now + hold_duration_ms(req.demand, req.priority, req.rid)
            self.active.append((complete_at, req.rid))
            return True, True, complete_at
        return False, False, -1

    def _record_grant(self, req, now, complete_at, basis, enqueued, used_burst):
        snap = self._snapshot(req)
        grant_basis = basis + ("|BURST" if used_burst else "|POOL")
        self.outcomes[req.rid] = Outcome(
            rid=req.rid,
            tid=req.tid,
            pid=req.pid,
            arrival=req.arrival,
            verdict=VERDICT_GRANT,
            reason=None,
            basis=grant_basis,
            decided_at=now,
            granted_at=now,
            completed_at=complete_at,
            wait_ms=now - req.arrival,
            tenant_quota_remaining=snap["tenant_quota_remaining"],
            tenant_burst_remaining=snap["tenant_burst_remaining"],
            inflight=snap["inflight"],
            pool_capacity_remaining=snap["pool_capacity_remaining"],
            demand=req.demand,
            enqueued=enqueued,
        )

    # ------------------------------------------------------------------ #
    # 到达判决（显式顺序）
    # ------------------------------------------------------------------ #
    def _admit(self, req: Request, now: int):
        self.requests_by_id[req.rid] = req
        self._now = now

        # 1) 租户累计配额（硬预算）
        if self.quota_remaining[req.tid] < req.demand:
            self._reject(req, now, REASON_TENANT_QUOTA, "ARRIVE|TENANT_QUOTA")
            return

        # 2) 在途并发上限：达到上限则排队，并预占租户配额
        if self.inflight[req.tid] >= self.tenants[req.tid].concurrency_cap:
            self.quota_remaining[req.tid] -= req.demand
            self._enqueue(req, now, "ARRIVE|CONCURRENCY_CAP")
            return

        # 3) 池常规容量足够：扣租户配额后常规授予
        if self.pool_remaining[req.pid] >= req.demand:
            self.quota_remaining[req.tid] -= req.demand
            ok, used_burst, complete_at = self._apply_grant(req, now)
            assert ok
            self._record_grant(req, now, complete_at, "ARRIVE", False, used_burst)
            return

        # 池常规容量不足 -> 拒绝分类
        # 4a) 无突发额度：池容量耗尽
        if self.burst_remaining[req.tid] == 0:
            self._reject(req, now, REASON_POOL_CAPACITY, "ARRIVE|POOL_CAPACITY")
            return
        # 4b) 有突发但不足以覆盖需求：突发额度不足
        if self.burst_remaining[req.tid] < req.demand or req.demand > self.pools[req.pid].capacity:
            self._reject(req, now, REASON_BURST, "ARRIVE|BURST_INSUFFICIENT")
            return
        # 4c) 突发足够：扣租户配额并以突发授予
        self.quota_remaining[req.tid] -= req.demand
        ok, used_burst, complete_at = self._apply_grant(req, now)
        assert ok and used_burst
        self._record_grant(req, now, complete_at, "ARRIVE", False, used_burst)

    def _reject(self, req, now, reason, basis):
        snap = self._snapshot(req)
        self.outcomes[req.rid] = Outcome(
            rid=req.rid,
            tid=req.tid,
            pid=req.pid,
            arrival=req.arrival,
            verdict=VERDICT_REJECT,
            reason=reason,
            basis=basis,
            decided_at=now,
            granted_at=None,
            completed_at=None,
            wait_ms=0,
            tenant_quota_remaining=snap["tenant_quota_remaining"],
            tenant_burst_remaining=snap["tenant_burst_remaining"],
            inflight=snap["inflight"],
            pool_capacity_remaining=snap["pool_capacity_remaining"],
            demand=req.demand,
            enqueued=False,
        )

    def _enqueue(self, req, now, basis):
        self.queues[req.pid].append(req.rid)
        self.queue_events.append(
            QueueEvent(
                rid=req.rid,
                tid=req.tid,
                pid=req.pid,
                kind="ENQUEUE",
                at_ms=now,
                wait_ms=0,
                basis=basis,
                detail=f"quota_reserved={req.demand}",
            )
        )
        snap = self._snapshot(req)
        self.outcomes[req.rid] = Outcome(
            rid=req.rid,
            tid=req.tid,
            pid=req.pid,
            arrival=req.arrival,
            verdict=VERDICT_QUEUE,
            reason=None,
            basis=basis,
            decided_at=now,
            granted_at=None,
            completed_at=None,
            wait_ms=0,
            tenant_quota_remaining=snap["tenant_quota_remaining"],
            tenant_burst_remaining=snap["tenant_burst_remaining"],
            inflight=snap["inflight"],
            pool_capacity_remaining=snap["pool_capacity_remaining"],
            demand=req.demand,
            enqueued=True,
        )

    # ------------------------------------------------------------------ #
    # 完成与稳定唤醒
    # ------------------------------------------------------------------ #
    def _drain_due_completions(self, now: int):
        """处理完成时刻 <= now 的授予，按 (completion_time, rid) 稳定释放。"""
        while True:
            due = [item for item in self.active if item[0] <= now]
            if not due:
                return
            complete_at, rid = min(due, key=lambda item: (item[0], item[1]))
            req = self.requests_by_id[rid]
            self.active.remove((complete_at, rid))
            self.inflight[req.tid] -= 1
            used_burst = self.outcomes[rid].basis.endswith("|BURST")
            if not used_burst:
                self.pool_remaining[req.pid] = min(
                    self.pools[req.pid].capacity,
                    self.pool_remaining[req.pid] + req.demand,
                )
            self._wake_pool(req.pid, complete_at)

    def _wake_pool(self, pid: str, now: int):
        """一次释放后按稳定序反复尝试唤醒，直到队首无法授予。"""
        while self.queues[pid]:
            candidates = [self.requests_by_id[rid] for rid in self.queues[pid]]
            candidates.sort(key=self._wake_key)
            head = candidates[0]
            ok, used_burst, complete_at = self._apply_grant(head, now)
            if not ok:
                self.queue_events.append(
                    QueueEvent(
                        rid=head.rid,
                        tid=head.tid,
                        pid=pid,
                        kind="STAY",
                        at_ms=now,
                        wait_ms=now - head.arrival,
                        basis="WAKE|REMAIN_QUEUED",
                        detail=self._stable_detail(head),
                    )
                )
                return
            self.queues[pid].remove(head.rid)
            wait_ms = now - head.arrival
            self._record_grant(head, now, complete_at, "WAKE", True, used_burst)
            self.queue_events.append(
                QueueEvent(
                    rid=head.rid,
                    tid=head.tid,
                    pid=pid,
                    kind="WAKE",
                    at_ms=now,
                    wait_ms=wait_ms,
                    basis="WAKE|STABLE_RULE",
                    detail=self._stable_detail(head),
                )
            )

    def _stable_detail(self, req: Request) -> str:
        return (
            f"pool_rank={self._pool_priority_rank(req.pid, req.tid)};"
            f"priority={req.priority};arrival={req.arrival};rid={req.rid}"
        )

    # ------------------------------------------------------------------ #
    # 驱动入口
    # ------------------------------------------------------------------ #
    def process_request(self, req: Request):
        """喂入一条请求（必须按规范序 (arrival, rid) 到达）。"""
        self._drain_due_completions(req.arrival)
        self._admit(req, req.arrival)

    def finalize(self):
        """流结束后排空所有活动授予并唤醒队列；仍无法授予的保持 QUEUE 终态。"""
        while self.active:
            next_complete = min(self.active, key=lambda item: (item[0], item[1]))[0]
            self._drain_due_completions(next_complete)
        for pid, rids in self.queues.items():
            for rid in list(rids):
                req = self.requests_by_id[rid]
                self.queue_events.append(
                    QueueEvent(
                        rid=rid,
                        tid=req.tid,
                        pid=pid,
                        kind="STAY",
                        at_ms=self._now,
                        wait_ms=self._now - req.arrival,
                        basis="FINAL|REMAIN_QUEUED",
                        detail=self._stable_detail(req),
                    )
                )

    def ordered_outcomes(self, request_order: list[Request]):
        return [self.outcomes[r.rid] for r in request_order]

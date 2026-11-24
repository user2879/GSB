"""每域 FIFO 排队协调器（引擎与真实回环服务共用的同构组件）。

职责：

* 维护每域等待队列，容量（每域租约上限 + 槽位）不足时入队；
* 严格 FIFO，队首不满足则后续事件必须等待，杜绝优先级插队；
* 到期回收 / 释放后泵取队首；
* 记录等待开始/结束的单调毫秒。

全局并发租约上限由跨域协调器在引擎侧或驱动侧以信号量约束，
本协调器只负责单域内的确定性准入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .engine import DomainArbiter, Lease, Outcome, WaitRecord
from .spec import InstanceSpec


@dataclass
class PendingAcquire:
    lease_id: str
    instance: InstanceSpec
    event_seq: int
    arrival_ms: int
    wait_start_ms: int


class DomainCoordinator:
    def __init__(self, arbiter: DomainArbiter, lease_ms_provider: Callable[[], int] | None = None) -> None:
        self.arbiter = arbiter
        self.queue: list[PendingAcquire] = []
        self.waits: list[WaitRecord] = []
        self.granted_by_seq: dict[int, Outcome] = {}
        self._lease_ms_provider = lease_ms_provider

    def submit_acquire(self, lease_id: str, instance: InstanceSpec,
                       event_seq: int, now: int) -> tuple[str, Outcome | None]:
        """返回 (status, outcome)；status 为 queued 表示已入队，结局稍后落账。"""
        if self.queue or not self._can_admit(instance):
            self.queue.append(PendingAcquire(
                lease_id=lease_id, instance=instance, event_seq=event_seq,
                arrival_ms=now, wait_start_ms=now,
            ))
            return "queued", None
        return self._grant(lease_id, instance, event_seq, now, wait_start=0)

    def _can_admit(self, instance: InstanceSpec) -> bool:
        arb = self.arbiter
        return len(arb.active) < arb.spec.lease_cap and len(arb.free_slots) >= instance.demand

    def _grant(self, lease_id: str, instance: InstanceSpec, event_seq: int,
               now: int, wait_start: int) -> tuple[str, Outcome]:
        lease_ms = self._lease_ms_provider() if self._lease_ms_provider else None
        status, lease, slots, expiry = self.arbiter.acquire(lease_id, instance, now, lease_ms)
        if status != "granted":
            return status, Outcome(
                seq=event_seq, arrival_ms=now, instance_id=instance.instance_id,
                domain_id=self.arbiter.spec.domain_id, action="acquire", status=status,
                lease_id="", version=0, slots="", expiry_ms=0,
                wait_start_ms=wait_start, wait_end_ms=now,
                wait_ms=max(0, now - wait_start), clock_ms=now,
            )
        assert lease is not None
        if wait_start:
            self.waits.append(WaitRecord(event_seq, instance.instance_id, wait_start, now))
        outcome = Outcome(
            seq=event_seq, arrival_ms=now, instance_id=instance.instance_id,
            domain_id=self.arbiter.spec.domain_id, action="acquire", status="granted",
            lease_id=lease.lease_id, version=lease.version,
            slots=",".join(str(slot) for slot in lease.slots), expiry_ms=lease.expiry_ms,
            wait_start_ms=wait_start, wait_end_ms=now,
            wait_ms=max(0, now - wait_start), clock_ms=now,
        )
        self.granted_by_seq[event_seq] = outcome
        return "granted", outcome

    def pump(self, now: int) -> list[Outcome]:
        """到期/释放后调用，按 FIFO 泵取所有可准入的队首。"""
        results: list[Outcome] = []
        while self.queue:
            head = self.queue[0]
            if not self._can_admit(head.instance):
                break
            self.queue.pop(0)
            _status, outcome = self._grant(
                head.lease_id, head.instance, head.event_seq, now, head.wait_start_ms)
            results.append(outcome)
        return results

    def drain_unmatched(self, now: int) -> list[Outcome]:
        """流末尾仍排队的任务落拒绝账。"""
        results: list[Outcome] = []
        while self.queue:
            head = self.queue.pop(0)
            self.waits.append(WaitRecord(head.event_seq, head.instance.instance_id,
                                         head.wait_start_ms, now))
            results.append(Outcome(
                seq=head.event_seq, arrival_ms=head.arrival_ms,
                instance_id=head.instance.instance_id,
                domain_id=self.arbiter.spec.domain_id, action="acquire",
                status="denied_quota", lease_id="", version=0, slots="", expiry_ms=0,
                wait_start_ms=head.wait_start_ms, wait_end_ms=now,
                wait_ms=max(0, now - head.wait_start_ms), clock_ms=now,
            ))
        return results

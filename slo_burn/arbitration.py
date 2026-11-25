"""并发槽位唯一归属裁决。

不变量：
  I1 同一槽位任一时刻最多一个持有者；
  I2 同一槽位的同一次释放只能归类为 expiry 或 revoke 之一，禁止双计；
  I3 每个归属变化记录事件序号、租约版本与裁决依据；
  I4 按任意服务处理顺序重放，裁决序列完全一致（事件先全序排序后裁决）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import LeaseError


class RecycleReason(str, Enum):
    GRANT = "grant"
    EXPIRY = "expiry"
    REVOKE = "revoke"


@dataclass(frozen=True)
class OwnershipEvent:
    """待裁决的原始事件（来自评估器/调度器）。"""

    observed_ms: int
    service_id: str
    slot_id: int
    lease_id: str
    version: int
    reason: RecycleReason
    source_order: int  # 同一毫秒内的稳定tie-breaker


@dataclass(frozen=True)
class Adjudication:
    event_seq: int
    observed_ms: int
    slot_id: int
    service_id: str
    lease_id: str
    version: int
    decision: str  # acquired | recycled-expiry | recycled-revoke | rejected-double-count
    basis: str
    prior_owner: str | None


class OwnershipArbiter:
    """确定性归属裁决状态机。"""

    def __init__(self) -> None:
        self._owner_by_slot: dict[int, str | None] = {}
        self._version_by_lease: dict[str, int] = {}
        self._recycled_leases: set[str] = set()
        self._event_seq = 0
        self.decisions: list[Adjudication] = []

    @staticmethod
    def total_order(events: list[OwnershipEvent]) -> list[OwnershipEvent]:
        """全序化：时间 -> 槽位 -> 服务 -> 原因(revoke 先于 expiry) -> 来源序号。"""
        reason_rank = {
            RecycleReason.GRANT: 0,
            RecycleReason.REVOKE: 1,
            RecycleReason.EXPIRY: 2,
        }
        return sorted(
            events,
            key=lambda event: (
                event.observed_ms,
                event.slot_id,
                event.service_id,
                reason_rank[event.reason],
                event.source_order,
            ),
        )

    def adjudicate_all(self, events: list[OwnershipEvent]) -> list[Adjudication]:
        for event in self.total_order(events):
            self._adjudicate(event)
        return list(self.decisions)

    def _adjudicate(self, event: OwnershipEvent) -> Adjudication:
        self._event_seq += 1
        prior_owner = self._owner_by_slot.get(event.slot_id)
        current_version = self._version_by_lease.get(event.lease_id)

        if event.reason == RecycleReason.GRANT:
            if prior_owner is not None and prior_owner != event.lease_id:
                incumbent_service = prior_owner.split("-slot")[0].replace("lease-", "", 1)
                if incumbent_service != event.service_id:
                    raise LeaseError(
                        "槽位已被其他服务持有，签发违反唯一归属",
                        slot_id=event.slot_id,
                        incumbent=prior_owner,
                        requester=event.lease_id,
                    )
                basis = "same-slot re-grant to newer lease version after recycle"
            else:
                basis = (
                    "grant-on-free-slot"
                    if prior_owner is None
                    else "grant-already-owner"
                )
            self._owner_by_slot[event.slot_id] = event.lease_id
            self._version_by_lease[event.lease_id] = event.version
            decision = Adjudication(
                self._event_seq,
                event.observed_ms,
                event.slot_id,
                event.service_id,
                event.lease_id,
                event.version,
                "acquired",
                basis,
                prior_owner,
            )
        else:
            if event.lease_id in self._recycled_leases:
                decision = Adjudication(
                    self._event_seq,
                    event.observed_ms,
                    event.slot_id,
                    event.service_id,
                    event.lease_id,
                    event.version,
                    "rejected-double-count",
                    "lease already recycled; expiry/revoke must not be double counted",
                    prior_owner,
                )
            elif prior_owner != event.lease_id:
                decision = Adjudication(
                    self._event_seq,
                    event.observed_ms,
                    event.slot_id,
                    event.service_id,
                    event.lease_id,
                    event.version,
                    "rejected-double-count",
                    "slot not owned by this lease (already released or never held)",
                    prior_owner,
                )
            else:
                label = (
                    "recycled-revoke"
                    if event.reason == RecycleReason.REVOKE
                    else "recycled-expiry"
                )
                basis = (
                    "explicit batch revocation precedes expiry in total order"
                    if event.reason == RecycleReason.REVOKE
                    else "monotonic clock passed expires_ms and no earlier recycle"
                )
                self._owner_by_slot[event.slot_id] = None
                self._recycled_leases.add(event.lease_id)
                decision = Adjudication(
                    self._event_seq,
                    event.observed_ms,
                    event.slot_id,
                    event.service_id,
                    event.lease_id,
                    event.version,
                    label,
                    basis,
                    prior_owner,
                )
        self.decisions.append(decision)
        return decision

    def slot_owners(self) -> dict[int, str | None]:
        return dict(self._owner_by_slot)

"""确定性租约仲裁引擎（全部路径的唯一事实来源）。

引擎只依赖可注入的单调时钟：

* :class:`~faultisolation.clock.LogicalClock` 用于确定性重放与独立校验；
* :class:`~faultisolation.clock.MonotonicClock` 用于真实回环仲裁服务。

设计要点：

* 租约是状态机，迁移仅允许 :data:`TRANSITIONS` 许可表中的边；
* 每个域维护空闲槽位栈与活跃租约映射，槽位一经回收才可能再分配，
  任何重复持有都会抛 ``E_ALIAS``；
* 全局并发上限与每域租约上限同时生效，耗尽即排队，FIFO 严格不插队；
* 同一到达时刻的仲裁顺序固定为 (priority 升序, instance_id 字典序)，可逐项复核；
* 所有结论字段均为整数/确定字符串，序列化后三条路径逐字节一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .clock import Clock, LogicalClock
from .errors import AliasError, LeaseError, StateError
from .spec import DomainSpec, Event, InstanceSpec, Session, SpecBundle

LEASE_STATES = ("granted", "renewed", "released", "expired")

# 显式许可表：(迁移前状态, 触发原因) -> 迁移后状态
TRANSITIONS: dict[tuple[str, str], str] = {
    ("<none>", "acquire_granted"): "granted",
    ("granted", "renew_ok"): "renewed",
    ("renewed", "renew_ok"): "renewed",
    ("granted", "release_ok"): "released",
    ("renewed", "release_ok"): "released",
    ("granted", "expiry_sweep"): "expired",
    ("renewed", "expiry_sweep"): "expired",
}
TERMINAL_STATES = frozenset(("released", "expired"))
ACTIVE_STATES = frozenset(("granted", "renewed"))
CANONICAL_EVENT_ORDER = "(arrival_ms 升序, priority 升序, instance_id 升序, seq 升序)"


@dataclass(frozen=True)
class TransitionRecord:
    lease_id: str
    from_state: str
    to_state: str
    reason: str
    clock_ms: int
    version: int


@dataclass(frozen=True)
class Outcome:
    """单事件确定性结论，字段顺序即账本与逐字节比对的列顺序。"""

    seq: int
    arrival_ms: int
    instance_id: str
    domain_id: str
    action: str
    status: str
    lease_id: str
    version: int
    slots: str
    expiry_ms: int
    wait_start_ms: int
    wait_end_ms: int
    wait_ms: int
    clock_ms: int

    def canonical_json(self) -> bytes:
        import json
        payload = {
            "seq": self.seq, "arrival_ms": self.arrival_ms,
            "instance_id": self.instance_id, "domain_id": self.domain_id,
            "action": self.action, "status": self.status,
            "lease_id": self.lease_id, "version": self.version,
            "slots": self.slots, "expiry_ms": self.expiry_ms,
            "wait_start_ms": self.wait_start_ms, "wait_end_ms": self.wait_end_ms,
            "wait_ms": self.wait_ms, "clock_ms": self.clock_ms,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class Lease:
    lease_id: str
    instance_id: str
    domain_id: str
    slots: tuple[int, ...]
    version: int
    state: str
    granted_ms: int
    expiry_ms: int
    last_renew_ms: int


@dataclass
class WaitRecord:
    seq: int
    instance_id: str
    start_ms: int
    end_ms: int

    @property
    def wait_ms(self) -> int:
        return self.end_ms - self.start_ms


class DomainArbiter:
    """单个故障域的租约状态机与槽位归属表（确定性，无线程原语）。"""

    def __init__(self, spec: DomainSpec, clock: Clock) -> None:
        self.spec = spec
        self.clock = clock
        self.free_slots: list[int] = list(range(spec.slots))
        self.active: dict[str, Lease] = {}
        self.slot_owner: dict[int, str] = {}
        self.history: dict[str, Lease] = {}
        self.transitions: list[TransitionRecord] = []

    def _record_transition(self, lease: Lease, from_state: str, reason: str,
                           to_state: str, clock_ms: int) -> None:
        allowed = TRANSITIONS.get((from_state, reason))
        if allowed != to_state:
            raise StateError(
                f"非法状态迁移 {from_state} --{reason}--> {to_state}",
                lease_id=lease.lease_id, allowed=allowed,
            )
        lease.state = to_state
        self.transitions.append(TransitionRecord(
            lease_id=lease.lease_id, from_state=from_state, to_state=to_state,
            reason=reason, clock_ms=clock_ms, version=lease.version,
        ))

    def sweep_expired(self, now: int | None = None) -> list[str]:
        """依据给定单调时刻回收到期租约，纯时钟比较，不使用任何 sleep。"""
        now = self.clock.now_ms() if now is None else now
        expired_ids: list[str] = []
        for lease in list(self.active.values()):
            if now >= lease.expiry_ms:
                self._reclaim(lease, reason="expiry_sweep", clock_ms=now)
                expired_ids.append(lease.lease_id)
        return expired_ids

    def _reclaim(self, lease: Lease, reason: str, clock_ms: int) -> None:
        from_state = lease.state
        for slot in lease.slots:
            owner = self.slot_owner.get(slot)
            if owner is not None and owner != lease.lease_id:
                raise AliasError(
                    f"槽位 {slot} 同时被 {owner} 与 {lease.lease_id} 持有",
                    lease_a=owner, lease_b=lease.lease_id, slot=slot,
                )
            self.slot_owner.pop(slot, None)
            self.free_slots.append(slot)
        self.free_slots.sort()
        self.active.pop(lease.lease_id, None)
        target = "expired" if reason == "expiry_sweep" else "released"
        self._record_transition(lease, from_state, reason, target, clock_ms)

    def acquire(self, lease_id: str, instance: InstanceSpec, now: int,
                lease_ms: int | None = None) -> tuple[str, Lease | None, tuple[int, ...], int]:
        """返回 (status, lease, slots, expiry_ms)。"""
        self.sweep_expired(now)
        if lease_id in self.active or lease_id in self.history:
            raise LeaseError(f"租约标识 {lease_id} 已存在，不能重复授予", lease_id=lease_id)
        if len(self.active) >= self.spec.lease_cap:
            return "denied_cap", None, (), 0
        if len(self.free_slots) < instance.demand:
            return "denied_quota", None, (), 0
        # 统一槽位分配规则：始终取当前最小的 demand 个空闲槽位（升序）
        slots = tuple(sorted(self.free_slots.pop(0) for _ in range(instance.demand)))
        for slot in slots:
            if slot in self.slot_owner:
                raise AliasError(
                    f"槽位 {slot} 已被 {self.slot_owner[slot]} 持有，无法授予 {lease_id}",
                    lease_a=self.slot_owner[slot], lease_b=lease_id, slot=slot,
                )
            self.slot_owner[slot] = lease_id
        duration = self.spec.lease_ms if lease_ms is None else lease_ms
        expiry = now + duration
        lease = Lease(
            lease_id=lease_id, instance_id=instance.instance_id, domain_id=self.spec.domain_id,
            slots=slots, version=1, state="granted", granted_ms=now,
            expiry_ms=expiry, last_renew_ms=now,
        )
        self.active[lease_id] = lease
        self.history[lease_id] = lease
        self._record_transition(lease, "<none>", "acquire_granted", "granted", now)
        return "granted", lease, slots, expiry

    def renew(self, lease_id: str, version: int, now: int | None = None) -> tuple[str, Lease | None, int]:
        now = self.clock.now_ms() if now is None else now
        self.sweep_expired(now)
        lease = self.active.get(lease_id)
        if lease is None:
            return "denied_lease", None, 0
        if lease.version != version:
            # 乐观版本不符：拒绝且不延长到期时刻
            raise LeaseError(
                f"续约版本不符：携带 {version}，当前 {lease.version}，租约不延长",
                lease_id=lease_id, presented=version, current=lease.version, expiry_ms=lease.expiry_ms,
            )
        from_state = lease.state
        lease.version += 1
        lease.expiry_ms = now + self.spec.lease_ms
        lease.last_renew_ms = now
        self._record_transition(lease, from_state, "renew_ok", "renewed", now)
        return "renewed", lease, lease.expiry_ms

    def release(self, lease_id: str, now: int | None = None) -> tuple[str, Lease | None]:
        self.sweep_expired()
        lease = self.active.get(lease_id)
        if lease is None:
            return "denied_lease", None
        now = self.clock.now_ms() if now is None else now
        self._reclaim(lease, reason="release_ok", clock_ms=now)
        return "released", lease

    def snapshot(self, now: int | None = None) -> dict:
        self.sweep_expired(now)
        return {
            "domain_id": self.spec.domain_id,
            "active": sorted(
                (
                    {
                        "lease_id": l.lease_id, "instance_id": l.instance_id,
                        "slots": list(l.slots), "version": l.version,
                        "state": l.state, "expiry_ms": l.expiry_ms,
                    }
                    for l in self.active.values()
                ),
                key=lambda item: item["lease_id"],
            ),
            "slot_owner": dict(sorted(self.slot_owner.items())),
            "free_slots": sorted(self.free_slots),
        }



def canonical_event_order(events, priority_of):
    """统一规范序：同一到达时刻内按 (priority, instance_id, seq) 排序，可逐项复核。"""
    return sorted(events, key=lambda e: (e.arrival_ms, priority_of[e.instance_id], e.instance_id, e.seq))


class Engine:
    """跨域确定性引擎：维护全局在途上限、各域协调器与完整账本数据。"""

    def __init__(self, bundle, clock=None):
        from .coordinator import DomainCoordinator
        self.bundle = bundle
        self.clock = clock or LogicalClock(0)
        self.session = bundle.session
        self.arbiters = {d.domain_id: DomainArbiter(d, self.clock) for d in bundle.domains}
        self.coordinators = {
            d.domain_id: DomainCoordinator(self.arbiters[d.domain_id]) for d in bundle.domains
        }
        self.instances = {i.instance_id: i for i in bundle.instances}
        self.priority_of = {i.instance_id: i.priority for i in bundle.instances}
        self.global_inflight = 0
        self.domain_inflight = {d.domain_id: 0 for d in bundle.domains}
        self.outcomes = []
        self.transitions = []
        self.waits = []
        self._outcome_index = {}

    def _new_lease_id(self, event, domain_id):
        # 由授予事件序号确定性派生，处理顺序变化也不会改变租约标识
        return f"L{event.seq:06d}:{domain_id}:{event.instance_id}"

    def _denied_outcome(self, event, domain_id, status, now):
        return Outcome(
            seq=event.seq, arrival_ms=event.arrival_ms, instance_id=event.instance_id,
            domain_id=domain_id, action=event.action, status=status,
            lease_id="", version=0, slots="", expiry_ms=0,
            wait_start_ms=0, wait_end_ms=0, wait_ms=0, clock_ms=now,
        )

    def _store(self, outcome):
        self.outcomes.append(outcome)
        self._outcome_index[outcome.seq] = outcome

    def _consume_arbiter_transitions(self, domain_id):
        arbiter = self.arbiters[domain_id]
        self.transitions.extend(arbiter.transitions)
        arbiter.transitions.clear()

    def _find_active_lease(self, instance_id, domain_id):
        for lease in self.arbiters[domain_id].active.values():
            if lease.instance_id == instance_id:
                return lease
        return None

    def _sweep_all(self, now):
        """全域到期回收（纯单调时钟比较），并泵取各域等待队列。"""
        for did, arbiter in self.arbiters.items():
            before = len(arbiter.active)
            arbiter.sweep_expired(now)
            reclaimed = before - len(arbiter.active)
            if reclaimed:
                self.global_inflight -= reclaimed
                self.domain_inflight[did] -= reclaimed
                self._consume_arbiter_transitions(did)
            for outcome in self.coordinators[did].pump(now):
                self.global_inflight += 1
                self.domain_inflight[did] += 1
                self._store(outcome)
            self.waits.extend(self.coordinators[did].waits)
            self.coordinators[did].waits.clear()
            self._consume_arbiter_transitions(did)

    def process_event(self, event):
        if isinstance(self.clock, LogicalClock):
            # 确定性模式：仲裁时刻恒等于事件到达时刻，与路径处理顺序无关
            now = event.arrival_ms
        else:
            now = self.clock.now_ms()
        self._sweep_all(now)
        domain_id = self.instances[event.instance_id].domain_id
        arbiter = self.arbiters[domain_id]
        coordinator = self.coordinators[domain_id]
        instance = self.instances[event.instance_id]

        if event.action == "acquire":
            lease_id = self._new_lease_id(event, domain_id)
            # 全局并发上限：受限时入队等待，FIFO 不允许优先级插队
            if self.global_inflight >= self.session.global_lease_cap and not coordinator.queue:
                from .coordinator import PendingAcquire
                coordinator.queue.append(PendingAcquire(
                    lease_id=lease_id, instance=instance, event_seq=event.seq,
                    arrival_ms=now, wait_start_ms=now))
                self._consume_arbiter_transitions(domain_id)
                return None
            status, outcome = coordinator.submit_acquire(lease_id, instance, event.seq, now)
            if status == "granted":
                self.global_inflight += 1
                self.domain_inflight[domain_id] += 1
                self._store(outcome)
            elif status == "queued":
                self._consume_arbiter_transitions(domain_id)
                return None
            else:
                self._store(outcome)
            self.waits.extend(coordinator.waits)
            coordinator.waits.clear()
            self._consume_arbiter_transitions(domain_id)
            return self._outcome_index.get(event.seq)

        if event.action == "renew":
            lease = self._find_active_lease(event.instance_id, domain_id)
            if lease is None:
                outcome = self._denied_outcome(event, domain_id, "denied_lease", now)
            else:
                status, renewed, expiry = arbiter.renew(lease.lease_id, lease.version, now)
                if status != "renewed":
                    outcome = self._denied_outcome(event, domain_id, status, now)
                else:
                    assert renewed is not None
                    outcome = Outcome(
                        seq=event.seq, arrival_ms=event.arrival_ms, instance_id=event.instance_id,
                        domain_id=domain_id, action="renew", status="renewed",
                        lease_id=renewed.lease_id, version=renewed.version,
                        slots=",".join(str(slot) for slot in renewed.slots), expiry_ms=expiry,
                        wait_start_ms=0, wait_end_ms=0, wait_ms=0, clock_ms=now,
                    )
            self._store(outcome)
            self._consume_arbiter_transitions(domain_id)
            return outcome

        # release
        lease = self._find_active_lease(event.instance_id, domain_id)
        if lease is None:
            outcome = self._denied_outcome(event, domain_id, "denied_lease", now)
        else:
            status, released = arbiter.release(lease.lease_id, now)
            if status != "released" or released is None:
                outcome = self._denied_outcome(event, domain_id, status, now)
            else:
                self.global_inflight -= 1
                self.domain_inflight[domain_id] -= 1
                outcome = Outcome(
                    seq=event.seq, arrival_ms=event.arrival_ms, instance_id=event.instance_id,
                    domain_id=domain_id, action="release", status="released",
                    lease_id=released.lease_id, version=released.version,
                    slots=",".join(str(slot) for slot in released.slots),
                    expiry_ms=released.expiry_ms,
                    wait_start_ms=0, wait_end_ms=0, wait_ms=0, clock_ms=now,
                )
        self._store(outcome)
        self._consume_arbiter_transitions(domain_id)
        for pumped in coordinator.pump(now):
            self.global_inflight += 1
            self.domain_inflight[domain_id] += 1
            self._store(pumped)
        self.waits.extend(coordinator.waits)
        coordinator.waits.clear()
        self._consume_arbiter_transitions(domain_id)
        return outcome

    def _run_ordered(self, ordered):
        for event in ordered:
            self.process_event(event)
        final_now = max(e.arrival_ms for e in self.bundle.events) if isinstance(
            self.clock, LogicalClock) else self.clock.now_ms()
        for did in sorted(self.coordinators):
            for outcome in self.coordinators[did].drain_unmatched(final_now):
                self._store(outcome)
            self.waits.extend(self.coordinators[did].waits)
            self.coordinators[did].waits.clear()
        return [self._outcome_index[event.seq] for event in canonical_event_order(
            self.bundle.events, self.priority_of)]

    def run_event_path(self, events=None):
        ordered = canonical_event_order(events or self.bundle.events, self.priority_of)
        return self._run_ordered(ordered)

    def run_batch_path(self, events=None):
        ordered = canonical_event_order(events or self.bundle.events, self.priority_of)
        batch_size = self.session.batch_size
        bucketed = []
        for start in range(0, len(ordered), batch_size):
            bucketed.extend(ordered[start:start + batch_size])
        return self._run_ordered(bucketed)

    def run_merge_path(self, events=None):
        ordered = canonical_event_order(events or self.bundle.events, self.priority_of)
        per_domain = {d.domain_id: 0 for d in self.bundle.domains}
        for event in ordered:
            per_domain[self.instances[event.instance_id].domain_id] += 1
        self._merge_groups = dict(sorted(per_domain.items()))
        return self._run_ordered(list(ordered))

"""Deterministic Serverless concurrency-lease simulator.

The engine deliberately keeps all ordering explicit.  No decision depends on
Python dictionary or set iteration order, so declaration traversal order cannot
change lease identifiers or grant order.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .errors import DeclError, QuotaError
from .fixedpoint import Ratio, quantize_ratio, rank_ratios
from .models import CallEvent, Declaration, SessionConfig, content_digest


ACTIVE = "ACTIVE"
STALE = "STALE"


@dataclass
class Lease:
    lease_id: str
    event_seq: int
    function_id: str
    grant_ms: int
    return_deadline_ms: int
    planned_release_ms: int
    concurrency: int
    remaining_occupancy_ms: int
    origin_version: int
    current_version: int
    status: str = ACTIVE
    release_ms: int | None = None
    release_disposition: str | None = None
    release_reason: str | None = None
    invalidation_reason: str | None = None
    loan_ids: list[str] = field(default_factory=list)
    queue_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "event_seq": self.event_seq,
            "function_id": self.function_id,
            "grant_ms": self.grant_ms,
            "return_deadline_ms": self.return_deadline_ms,
            "planned_release_ms": self.planned_release_ms,
            "concurrency": self.concurrency,
            "remaining_occupancy_ms": self.remaining_occupancy_ms,
            "origin_version": self.origin_version,
            "current_version": self.current_version,
            "status": self.status,
            "release_ms": self.release_ms,
            "release_disposition": self.release_disposition,
            "release_reason": self.release_reason,
            "invalidation_reason": self.invalidation_reason,
            "loan_ids": list(self.loan_ids),
            "queue_id": self.queue_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Lease":
        return cls(**raw)


@dataclass
class QueueEntry:
    queue_id: str
    queue_seq: int
    event_seq: int
    function_id: str
    demand: int
    remaining_occupancy_ms: int
    arrived_ms: int
    enqueued_ms: int
    not_before_ms: int
    source: str
    declaration_version: int
    status: str = "PENDING"
    lease_id: str | None = None
    resolved_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "queue_seq": self.queue_seq,
            "event_seq": self.event_seq,
            "function_id": self.function_id,
            "demand": self.demand,
            "remaining_occupancy_ms": self.remaining_occupancy_ms,
            "arrived_ms": self.arrived_ms,
            "enqueued_ms": self.enqueued_ms,
            "not_before_ms": self.not_before_ms,
            "source": self.source,
            "declaration_version": self.declaration_version,
            "status": self.status,
            "lease_id": self.lease_id,
            "resolved_ms": self.resolved_ms,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QueueEntry":
        return cls(**raw)


@dataclass
class Loan:
    loan_id: str
    lease_id: str
    lender_id: str
    borrower_id: str
    amount: int
    created_ms: int
    return_deadline_ms: int
    origin_version: int
    current_version: int
    status: str = "ACTIVE"
    returned_ms: int | None = None
    disposition: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "loan_id": self.loan_id,
            "lease_id": self.lease_id,
            "lender_id": self.lender_id,
            "borrower_id": self.borrower_id,
            "amount": self.amount,
            "created_ms": self.created_ms,
            "return_deadline_ms": self.return_deadline_ms,
            "origin_version": self.origin_version,
            "current_version": self.current_version,
            "status": self.status,
            "returned_ms": self.returned_ms,
            "disposition": self.disposition,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Loan":
        return cls(**raw)


@dataclass
class Preemption:
    preemption_id: str
    beneficiary_event_seq: int
    beneficiary_function: str
    victim_lease_id: str
    victim_function: str
    occupied_concurrency: int
    scheduled_ms: int
    suspend_at_ms: int
    resume_not_before_ms: int
    status: str = "SCHEDULED"
    suspended_ms: int | None = None
    requeued_queue_id: str | None = None
    reason: str = "HIGHER_PRIORITY_QUOTA_SHORTAGE"
    declaration_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "preemption_id": self.preemption_id,
            "beneficiary_event_seq": self.beneficiary_event_seq,
            "beneficiary_function": self.beneficiary_function,
            "victim_lease_id": self.victim_lease_id,
            "victim_function": self.victim_function,
            "occupied_concurrency": self.occupied_concurrency,
            "scheduled_ms": self.scheduled_ms,
            "suspend_at_ms": self.suspend_at_ms,
            "resume_not_before_ms": self.resume_not_before_ms,
            "status": self.status,
            "suspended_ms": self.suspended_ms,
            "requeued_queue_id": self.requeued_queue_id,
            "reason": self.reason,
            "declaration_version": self.declaration_version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Preemption":
        value = dict(raw)
        value.setdefault("declaration_version", 0)
        return cls(**value)


@dataclass
class ReleaseRecord:
    lease_id: str
    event_seq: int
    function_id: str
    release_ms: int
    concurrency: int
    disposition: str
    reason: str
    remaining_occupancy_ms: int
    requeued_queue_id: str | None
    returned_loan_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "event_seq": self.event_seq,
            "function_id": self.function_id,
            "release_ms": self.release_ms,
            "concurrency": self.concurrency,
            "disposition": self.disposition,
            "reason": self.reason,
            "remaining_occupancy_ms": self.remaining_occupancy_ms,
            "requeued_queue_id": self.requeued_queue_id,
            "returned_loan_ids": list(self.returned_loan_ids),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReleaseRecord":
        return cls(**raw)


@dataclass(frozen=True)
class GrantPlan:
    loan_plan: tuple[tuple[str, int, int], ...]


@dataclass
class PlacementDecision:
    granted: bool
    reason: str
    global_free: int
    function_free: int
    shortage: int
    plan: GrantPlan | None = None


class LeaseEngine:
    def __init__(self, declaration: Declaration, session: SessionConfig) -> None:
        self.declaration = declaration
        self.session = session
        self.clock_ms = 0
        self.global_used = 0
        self.function_used: dict[str, int] = {
            function_id: 0 for function_id in sorted(declaration.functions)
        }
        self.leases: dict[str, Lease] = {}
        self.queue: dict[str, QueueEntry] = {}
        self.loans: dict[str, Loan] = {}
        self.preemptions: dict[str, Preemption] = {}
        self.releases: list[ReleaseRecord] = []
        self.invalidations: list[dict[str, Any]] = []
        self.latest_watermarks: list[dict[str, Any]] = []
        self.queue_seq = 0
        self.grant_counts: dict[int, int] = {}
        self.events_index: dict[int, CallEvent] = {}
        self.current_ops: list[dict[str, Any]] = []
        self._heap: list[tuple[int, int, int, str, str | None]] = []
        self._schedule: dict[str, tuple[int, int, str, str | None, int]] = {}
        self._schedule_token = 0
        self._inject_quota = False
        self.batch_id: str | None = None

    def begin_batch(self, batch_id: str) -> None:
        self.batch_id = batch_id
        self.current_ops = []

    def run_batch(self, events: Sequence[CallEvent]) -> dict[str, Any]:
        if not events:
            raise ValueError("batch must contain at least one event")
        ordered = sorted(events, key=lambda event: event.event_seq)
        arrival = ordered[0].arrival_ms
        if any(event.arrival_ms != arrival for event in ordered):
            raise ValueError("a batch may only contain events with one arrival_ms")
        if self.batch_id is None:
            self.begin_batch(self.make_batch_id(ordered))

        for event in ordered:
            self.events_index[event.event_seq] = event

        self._advance_releases_before(arrival)
        self.clock_ms = arrival
        self._process_releases_at(arrival, dispatch_after=False)

        for event in ordered:
            self._apply_event(event)

        self._dispatch_queue(arrival)
        self._process_releases_at(arrival, dispatch_after=True)
        self.clock_ms = arrival
        self.latest_watermarks = self.watermarks()
        return self.finish_batch(ordered)

    def finish_batch(self, events: Sequence[CallEvent]) -> dict[str, Any]:
        state = self.to_state_dict()
        state_digest = content_digest(state)
        payload = {
            "format_version": 1,
            "batch_id": self.batch_id,
            "declaration_version": self.declaration.declaration_version,
            "declaration_digest": self.declaration.digest,
            "scale": self.session.watermark_scale,
            "start_event_seq": min(event.event_seq for event in events),
            "end_event_seq": max(event.event_seq for event in events),
            "cursor_event_seq": max(event.event_seq for event in events),
            "arrival_ms": events[0].arrival_ms,
            "events": [event.as_dict() for event in events],
            "operations": list(self.current_ops),
            "watermarks": list(self.latest_watermarks),
            "state_digest": state_digest,
        }
        self.current_ops = []
        self.batch_id = None
        return payload

    def make_batch_id(self, events: Sequence[CallEvent]) -> str:
        start = min(event.event_seq for event in events)
        end = max(event.event_seq for event in events)
        return (
            f"B{self.declaration.declaration_version:04d}-"
            f"{start:08d}-{end:08d}"
        )

    def advance_to(self, target_ms: int) -> None:
        if target_ms < self.clock_ms:
            raise ValueError("clock cannot move backwards")
        self._advance_releases_before(target_ms)
        self._process_releases_at(target_ms, dispatch_after=True)
        self.clock_ms = target_ms

    def release_lease(
        self,
        lease_id: str,
        at_ms: int,
        *,
        reason: str = "EXPLICIT_RELEASE",
        requeue: bool = False,
    ) -> ReleaseRecord:
        self.advance_to(at_ms)
        lease = self.leases.get(lease_id)
        if lease is None:
            raise KeyError(lease_id)
        if lease.status != ACTIVE:
            raise ValueError(f"lease is not active: {lease_id}")
        not_before = max(at_ms + 1, at_ms) if requeue else None
        release = self._release_lease(
            lease,
            at_ms,
            disposition="EXPLICIT_RELEASE",
            reason=reason,
            requeue=requeue,
            requeue_not_before=not_before,
            source="EXPLICIT_RELEASE",
        )
        self._dispatch_queue(at_ms)
        self._process_releases_at(at_ms, dispatch_after=True)
        return release

    def set_quota_injection(self, enabled: bool) -> None:
        self._inject_quota = enabled

    def _apply_event(self, event: CallEvent) -> None:
        function = self.declaration.functions[event.function_id]
        decision = self._evaluate_placement(
            function_id=event.function_id,
            demand=event.concurrency_demand,
            now=event.arrival_ms,
        )
        if decision.granted:
            lease = self._grant(
                event_seq=event.event_seq,
                function_id=event.function_id,
                demand=event.concurrency_demand,
                remaining_occupancy_ms=event.expected_occupancy_ms,
                granted_ms=event.arrival_ms,
                plan=decision.plan,
                queue_entry=None,
            )
            self._op(
                "APPLICATION",
                event_seq=event.event_seq,
                function_id=event.function_id,
                at_ms=event.arrival_ms,
                demand=event.concurrency_demand,
                outcome="GRANTED",
                lease_id=lease.lease_id,
            )
            return

        queue_entry = self._enqueue(
            event_seq=event.event_seq,
            function_id=event.function_id,
            demand=event.concurrency_demand,
            remaining_occupancy_ms=event.expected_occupancy_ms,
            arrived_ms=event.arrival_ms,
            enqueued_ms=event.arrival_ms,
            not_before_ms=event.arrival_ms,
            source="INITIAL",
        )
        self._op(
            "APPLICATION",
            event_seq=event.event_seq,
            function_id=event.function_id,
            at_ms=event.arrival_ms,
            demand=event.concurrency_demand,
            outcome="QUEUED",
            queue_id=queue_entry.queue_id,
            reason=decision.reason,
            global_free=decision.global_free,
            function_free=decision.function_free,
        )
        if decision.reason in {"FUNCTION_QUOTA", "BORROW_CAPACITY"}:
            self._schedule_preemptions(queue_entry, decision.shortage, event.arrival_ms)

    def _evaluate_placement(
        self, *, function_id: str, demand: int, now: int
    ) -> PlacementDecision:
        function = self.declaration.functions[function_id]
        global_free = self.declaration.global_concurrency - self.global_used
        if demand > global_free:
            return PlacementDecision(
                granted=False,
                reason="GLOBAL_BUDGET",
                global_free=global_free,
                function_free=0,
                shortage=demand,
            )

        capacity = self._effective_capacity(function_id)
        function_free = max(0, capacity - self.function_used[function_id])
        shortage = max(0, demand - function_free)
        if shortage == 0:
            return PlacementDecision(
                granted=True,
                reason="OWN_QUOTA",
                global_free=global_free,
                function_free=function_free,
                shortage=0,
                plan=GrantPlan(loan_plan=()),
            )

        borrowed: list[tuple[str, int, int]] = []
        remaining_shortage = shortage
        for relation in self._borrow_relations(function_id):
            lender_id = relation.lower
            lender = self.declaration.functions[lender_id]
            if lender.priority >= function.priority:
                continue
            available = self._lender_surplus(lender_id)
            if available <= 0:
                continue
            amount = min(remaining_shortage, available)
            if amount:
                borrowed.append(
                    (lender_id, amount, now + relation.return_deadline_ms)
                )
                remaining_shortage -= amount
            if remaining_shortage == 0:
                break
        if remaining_shortage == 0:
            return PlacementDecision(
                granted=True,
                reason="BORROWED_QUOTA",
                global_free=global_free,
                function_free=function_free,
                shortage=0,
                plan=GrantPlan(loan_plan=tuple(borrowed)),
            )
        return PlacementDecision(
            granted=False,
            reason="FUNCTION_QUOTA" if not borrowed else "BORROW_CAPACITY",
            global_free=global_free,
            function_free=function_free,
            shortage=remaining_shortage,
        )

    def _grant(
        self,
        *,
        event_seq: int,
        function_id: str,
        demand: int,
        remaining_occupancy_ms: int,
        granted_ms: int,
        plan: GrantPlan | None,
        queue_entry: QueueEntry | None,
    ) -> Lease:
        function = self.declaration.functions[function_id]
        if plan is None:
            raise RuntimeError("grant requires a placement plan")
        grant_count = self.grant_counts.get(event_seq, 0)
        self.grant_counts[event_seq] = grant_count + 1
        lease_id = (
            f"L{event_seq:08d}"
            if grant_count == 0
            else f"L{event_seq:08d}R{grant_count:02d}"
        )

        if self.global_used + demand > self.declaration.global_concurrency:
            raise QuotaError(
                "global_concurrency",
                "grant would exceed global concurrency budget",
                event_seq=event_seq,
                lease_id=lease_id,
                used=self.global_used,
                demand=demand,
                budget=self.declaration.global_concurrency,
            )
        borrowed_total = sum(amount for _, amount, _ in (plan.loan_plan or ()))
        if self._effective_capacity(function_id) - demand + borrowed_total < self.function_used[function_id]:
            raise QuotaError(
                "max_concurrency",
                "grant would exceed function concurrency quota",
                event_seq=event_seq,
                lease_id=lease_id,
                used=self.function_used[function_id],
                demand=demand,
                capacity=self._effective_capacity(function_id),
                borrowed=borrowed_total,
            )
        if self._inject_quota:
            raise QuotaError(
                "quota_injection",
                "injected grant-over-limit fault",
                event_seq=event_seq,
                lease_id=lease_id,
                used=self.global_used,
                demand=demand,
                budget=self.declaration.global_concurrency,
            )

        hard_deadline = granted_ms + function.lease_ms
        completion_deadline = granted_ms + remaining_occupancy_ms
        planned_release = min(hard_deadline, completion_deadline)
        lease = Lease(
            lease_id=lease_id,
            event_seq=event_seq,
            function_id=function_id,
            grant_ms=granted_ms,
            return_deadline_ms=hard_deadline,
            planned_release_ms=planned_release,
            concurrency=demand,
            remaining_occupancy_ms=remaining_occupancy_ms,
            origin_version=self.declaration.declaration_version,
            current_version=self.declaration.declaration_version,
            queue_id=queue_entry.queue_id if queue_entry else None,
        )
        self.leases[lease_id] = lease
        self.global_used += demand
        self.function_used[function_id] += demand

        for preemption in sorted(
            self.preemptions.values(), key=lambda item: item.preemption_id
        ):
            if (
                preemption.beneficiary_event_seq == event_seq
                and preemption.status == "SCHEDULED"
            ):
                preemption.status = "CANCELLED_BENEFICIARY_GRANTED"
                self._op("PREEMPT_CANCEL", preemption=preemption.as_dict())
                victim = self.leases.get(preemption.victim_lease_id)
                if victim is not None and victim.status == ACTIVE:
                    self._restore_natural_schedule(victim)

        active_loans: list[Loan] = []
        for index, (lender_id, amount, deadline) in enumerate(plan.loan_plan, start=1):
            loan_id = f"B{lease_id}-{index:02d}"
            loan = Loan(
                loan_id=loan_id,
                lease_id=lease_id,
                lender_id=lender_id,
                borrower_id=function_id,
                amount=amount,
                created_ms=granted_ms,
                return_deadline_ms=deadline,
                origin_version=self.declaration.declaration_version,
                current_version=self.declaration.declaration_version,
            )
            self.loans[loan_id] = loan
            lease.loan_ids.append(loan_id)
            active_loans.append(loan)

        self._schedule_initial_release(lease, active_loans)
        self._op("GRANT", lease=lease.as_dict())
        for loan in active_loans:
            self._op("BORROW", loan=loan.as_dict())
        if queue_entry is not None:
            queue_entry.status = "GRANTED"
            queue_entry.lease_id = lease_id
            queue_entry.resolved_ms = granted_ms
            self._op("QUEUE_RESOLVED", queue=queue_entry.as_dict())
        return lease

    def _enqueue(
        self,
        *,
        event_seq: int,
        function_id: str,
        demand: int,
        remaining_occupancy_ms: int,
        arrived_ms: int,
        enqueued_ms: int,
        not_before_ms: int,
        source: str,
    ) -> QueueEntry:
        self.queue_seq += 1
        entry = QueueEntry(
            queue_id=f"Q{self.queue_seq:08d}",
            queue_seq=self.queue_seq,
            event_seq=event_seq,
            function_id=function_id,
            demand=demand,
            remaining_occupancy_ms=remaining_occupancy_ms,
            arrived_ms=arrived_ms,
            enqueued_ms=enqueued_ms,
            not_before_ms=not_before_ms,
            source=source,
            declaration_version=self.declaration.declaration_version,
        )
        self.queue[entry.queue_id] = entry
        self._op("QUEUE", queue=entry.as_dict())
        return entry

    def _dispatch_queue(self, now: int) -> None:
        while True:
            pending = [
                entry
                for entry in self.queue.values()
                if entry.status == "PENDING" and entry.not_before_ms <= now
            ]
            pending.sort(
                key=lambda entry: (
                    -self.declaration.functions[entry.function_id].priority,
                    entry.enqueued_ms,
                    entry.queue_seq,
                    entry.event_seq,
                )
            )
            granted_any = False
            for entry in pending:
                decision = self._evaluate_placement(
                    function_id=entry.function_id,
                    demand=entry.demand,
                    now=now,
                )
                if decision.granted:
                    self._grant(
                        event_seq=entry.event_seq,
                        function_id=entry.function_id,
                        demand=entry.demand,
                        remaining_occupancy_ms=entry.remaining_occupancy_ms,
                        granted_ms=now,
                        plan=decision.plan,
                        queue_entry=entry,
                    )
                    self._op(
                        "QUEUE_GRANT",
                        event_seq=entry.event_seq,
                        function_id=entry.function_id,
                        at_ms=now,
                        queue_id=entry.queue_id,
                        lease_id=entry.lease_id,
                    )
                    granted_any = True
                elif decision.reason in {"FUNCTION_QUOTA", "BORROW_CAPACITY"}:
                    self._schedule_preemptions(entry, decision.shortage, now)
            if not granted_any:
                return

    def _schedule_initial_release(self, lease: Lease, loans: Iterable[Loan]) -> None:
        candidates: list[tuple[int, int, str, str | None]] = [
            (
                lease.planned_release_ms,
                2
                if lease.planned_release_ms <= lease.return_deadline_ms
                else 3,
                "CALL_COMPLETED"
                if lease.planned_release_ms <= lease.return_deadline_ms
                else "LEASE_EXPIRED",
                None,
            )
        ]
        for loan in loans:
            candidates.append((loan.return_deadline_ms, 0, "BORROW_RETURN_DUE", None))
        when, rank, kind, preemption_id = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        self._schedule_lease_release(lease, when, kind, rank, preemption_id)

    def _restore_natural_schedule(self, lease: Lease) -> None:
        if lease.status != ACTIVE:
            return
        self._schedule.pop(lease.lease_id, None)
        loans = [
            self.loans[loan_id]
            for loan_id in lease.loan_ids
            if loan_id in self.loans and self.loans[loan_id].status == "ACTIVE"
        ]
        self._schedule_initial_release(lease, loans)

    def _schedule_lease_release(
        self,
        lease: Lease,
        when: int,
        kind: str,
        rank: int,
        preemption_id: str | None,
    ) -> None:
        existing = self._schedule.get(lease.lease_id)
        candidate = (when, rank, kind, preemption_id, self._schedule_token + 1)
        if existing is not None and (existing[0], existing[1]) <= (when, rank):
            return
        self._schedule_token += 1
        self._schedule[lease.lease_id] = (
            when,
            rank,
            kind,
            preemption_id,
            self._schedule_token,
        )
        heapq.heappush(
            self._heap,
            (when, rank, self._schedule_token, lease.lease_id, preemption_id),
        )

    def _advance_releases_before(self, target: int) -> None:
        while self._heap and self._heap[0][0] < target:
            when = self._heap[0][0]
            self._process_releases_at(when, dispatch_after=True)
            self.clock_ms = when

    def _process_releases_at(self, when: int, *, dispatch_after: bool) -> None:
        while self._heap and self._heap[0][0] == when:
            group: list[tuple[int, int, int, str, str | None]] = []
            while self._heap and self._heap[0][0] == when:
                group.append(heapq.heappop(self._heap))
            group.sort(key=lambda item: (item[1], item[3], item[2]))
            for _, _, token, lease_id, preemption_id in group:
                current = self._schedule.get(lease_id)
                if current is None or current[4] != token:
                    continue
                lease = self.leases.get(lease_id)
                if lease is None or lease.status != ACTIVE:
                    continue
                kind = current[2]
                if kind == "PREEMPTED":
                    preemption = self.preemptions.get(preemption_id or "")
                    if preemption is None or preemption.status != "SCHEDULED":
                        self._restore_natural_schedule(lease)
                        continue
                    requeue_not_before = preemption.resume_not_before_ms
                    release = self._release_lease(
                        lease,
                        when,
                        disposition="PREEMPTED",
                        reason="HIGHER_PRIORITY_PREEMPTION_BOUNDARY",
                        requeue=True,
                        requeue_not_before=requeue_not_before,
                        source="PREEMPTION",
                    )
                    preemption.status = "SUSPENDED"
                    preemption.suspended_ms = when
                    preemption.requeued_queue_id = release.requeued_queue_id
                    self._op("PREEMPT_SUSPEND", preemption=preemption.as_dict())
                elif kind == "BORROW_RETURN_DUE":
                    self._release_lease(
                        lease,
                        when,
                        disposition="BORROW_RETURN_DUE",
                        reason="BORROW_RETURN_DEADLINE",
                        requeue=True,
                        requeue_not_before=max(when + 1, when),
                        source="BORROW_RETURN",
                    )
                elif kind == "LEASE_EXPIRED":
                    self._release_lease(
                        lease,
                        when,
                        disposition="LEASE_EXPIRED",
                        reason="LEASE_DEADLINE",
                        requeue=True,
                        requeue_not_before=max(when + 1, when),
                        source="LEASE_EXPIRY",
                    )
                else:
                    self._release_lease(
                        lease,
                        when,
                        disposition="CALL_COMPLETED",
                        reason="EXPECTED_OCCUPANCY_COMPLETE",
                        requeue=False,
                        requeue_not_before=None,
                        source="COMPLETION",
                    )
        if dispatch_after:
            self._dispatch_queue(when)

    def _release_lease(
        self,
        lease: Lease,
        when: int,
        *,
        disposition: str,
        reason: str,
        requeue: bool,
        requeue_not_before: int | None,
        source: str,
    ) -> ReleaseRecord:
        if lease.status != ACTIVE:
            raise RuntimeError(f"double release attempted for {lease.lease_id}")
        elapsed = max(0, when - lease.grant_ms)
        remaining = (
            0
            if disposition == "CALL_COMPLETED"
            else max(0, lease.remaining_occupancy_ms - elapsed)
        )
        lease.status = "RELEASED"
        lease.release_ms = when
        lease.release_disposition = disposition
        lease.release_reason = reason
        self.global_used -= lease.concurrency
        self.function_used[lease.function_id] -= lease.concurrency
        if self.global_used < 0 or self.function_used[lease.function_id] < 0:
            raise RuntimeError("concurrency accounting underflow")

        returned: list[str] = []
        for loan_id in lease.loan_ids:
            loan = self.loans.get(loan_id)
            if loan is None or loan.status != "ACTIVE":
                continue
            loan.status = "RETURNED"
            loan.returned_ms = when
            loan.reason = reason
            if when < loan.return_deadline_ms:
                loan.disposition = "RETURNED_EARLY"
            elif when == loan.return_deadline_ms:
                loan.disposition = "RETURNED_AT_DEADLINE"
            else:
                loan.disposition = "RETURNED_LATE"
            returned.append(loan_id)
            self._op("LOAN_RETURN", loan=loan.as_dict())

        if disposition != "PREEMPTED":
            for preemption in sorted(
                self.preemptions.values(), key=lambda item: item.preemption_id
            ):
                if (
                    preemption.victim_lease_id == lease.lease_id
                    and preemption.status == "SCHEDULED"
                ):
                    preemption.status = "CANCELLED_LEASE_RELEASED"
                    self._op("PREEMPT_CANCEL", preemption=preemption.as_dict())

        requeued_queue_id: str | None = None
        if requeue and remaining > 0:
            entry = self._enqueue(
                event_seq=lease.event_seq,
                function_id=lease.function_id,
                demand=lease.concurrency,
                remaining_occupancy_ms=remaining,
                arrived_ms=self._event_arrival_ms(lease.event_seq),
                enqueued_ms=when,
                not_before_ms=requeue_not_before if requeue_not_before is not None else when,
                source=source,
            )
            requeued_queue_id = entry.queue_id
            self._op(
                "REQUEUE",
                event_seq=lease.event_seq,
                function_id=lease.function_id,
                at_ms=when,
                queue_id=entry.queue_id,
                remaining_occupancy_ms=remaining,
                not_before_ms=entry.not_before_ms,
                source=source,
            )

        record = ReleaseRecord(
            lease_id=lease.lease_id,
            event_seq=lease.event_seq,
            function_id=lease.function_id,
            release_ms=when,
            concurrency=lease.concurrency,
            disposition=disposition,
            reason=reason,
            remaining_occupancy_ms=remaining,
            requeued_queue_id=requeued_queue_id,
            returned_loan_ids=returned,
        )
        self.releases.append(record)
        self._op("RELEASE", release=record.as_dict())
        return record

    def _schedule_preemptions(
        self, queue_entry: QueueEntry, shortage: int, now: int
    ) -> None:
        if shortage <= 0:
            return
        already_reserved = sum(
            item.occupied_concurrency
            for item in self.preemptions.values()
            if item.status == "SCHEDULED"
            and item.beneficiary_event_seq == queue_entry.event_seq
        )
        shortage -= already_reserved
        if shortage <= 0:
            return
        function = self.declaration.functions[queue_entry.function_id]
        relations = [
            relation
            for relation in self._borrow_relations(queue_entry.function_id)
            if self.declaration.functions[relation.lower].priority < function.priority
        ]
        if not relations:
            return
        lower_ids = {relation.lower for relation in relations}
        reserved = {
            item.victim_lease_id
            for item in self.preemptions.values()
            if item.status == "SCHEDULED"
        }
        candidates: list[tuple[int, int, int, str, Lease]] = []
        for lease in self.leases.values():
            if lease.status != ACTIVE or lease.function_id not in lower_ids:
                continue
            if lease.lease_id in reserved:
                continue
            if lease.event_seq == queue_entry.event_seq:
                continue
            grid_boundary = (
                (now // self.declaration.preemption_checkpoint_ms) + 1
            ) * self.declaration.preemption_checkpoint_ms
            boundary = min(grid_boundary, lease.planned_release_ms)
            if boundary <= now or lease.planned_release_ms <= boundary:
                continue
            existing_schedule = self._schedule.get(lease.lease_id)
            if existing_schedule is not None and existing_schedule[0] <= boundary:
                continue
            relation = self.declaration.relation(
                queue_entry.function_id, lease.function_id
            )
            if relation is None:
                continue
            victim_priority = self.declaration.functions[lease.function_id].priority
            candidates.append(
                (victim_priority, boundary, lease.concurrency, lease.lease_id, lease)
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[3]))
        selected: list[Lease] = []
        selected_amount = 0
        for _, _, concurrency, _, lease in candidates:
            selected.append(lease)
            selected_amount += concurrency
            if selected_amount >= shortage:
                break
        for lease in selected:
            relation = self.declaration.relation(
                queue_entry.function_id, lease.function_id
            )
            assert relation is not None
            boundary = min(
                ((now // self.declaration.preemption_checkpoint_ms) + 1)
                * self.declaration.preemption_checkpoint_ms,
                lease.planned_release_ms,
            )
            resume_after = boundary + max(1, relation.return_deadline_ms)
            preemption_id = (
                f"P{self.declaration.declaration_version:04d}-"
                f"{queue_entry.event_seq:08d}-{lease.lease_id}"
            )
            if preemption_id in self.preemptions:
                continue
            preemption = Preemption(
                preemption_id=preemption_id,
                beneficiary_event_seq=queue_entry.event_seq,
                beneficiary_function=queue_entry.function_id,
                victim_lease_id=lease.lease_id,
                victim_function=lease.function_id,
                occupied_concurrency=lease.concurrency,
                scheduled_ms=now,
                suspend_at_ms=boundary,
                resume_not_before_ms=resume_after,
                declaration_version=self.declaration.declaration_version,
            )
            self.preemptions[preemption_id] = preemption
            self._schedule_lease_release(
                lease, boundary, "PREEMPTED", 1, preemption_id
            )
            self._op("PREEMPT_SCHEDULE", preemption=preemption.as_dict())

    def apply_declaration(self, new_declaration: Declaration, at_ms: int) -> bool:
        old = self.declaration
        old_version = old.declaration_version
        new_version = new_declaration.declaration_version
        if new_version < old_version:
            raise DeclError(
                "declaration_version",
                "declaration version must increase monotonically",
                previous=old_version,
                actual=new_version,
            )
        if new_version == old_version:
            if new_declaration.digest != old.digest:
                raise DeclError(
                    "declaration_version",
                    "different declaration content cannot reuse a version",
                    version=new_version,
                )
            return False

        changed_functions = {
            function_id
            for function_id in set(old.functions) | set(new_declaration.functions)
            if old.functions.get(function_id) != new_declaration.functions.get(function_id)
        }
        old_edges = {
            (edge.higher, edge.lower): edge.return_deadline_ms
            for edge in old.preemptions
        }
        new_edges = {
            (edge.higher, edge.lower): edge.return_deadline_ms
            for edge in new_declaration.preemptions
        }
        changed_edges = {
            edge
            for edge in set(old_edges) | set(new_edges)
            if old_edges.get(edge) != new_edges.get(edge)
        }
        affected_functions = set(changed_functions)
        for higher, lower in changed_edges:
            affected_functions.add(higher)
            affected_functions.add(lower)
        global_changed = old.global_concurrency != new_declaration.global_concurrency
        if global_changed:
            affected_functions = set(new_declaration.functions) | set(old.functions)

        affected_loans = {
            loan_id
            for loan_id, loan in self.loans.items()
            if loan.status == "ACTIVE"
            and (
                global_changed
                or loan.lender_id in affected_functions
                or loan.borrower_id in affected_functions
            )
        }
        affected_lease_ids = {
            lease_id
            for lease_id, lease in self.leases.items()
            if lease.status == ACTIVE
            and (
                global_changed
                or lease.function_id in affected_functions
                or any(
                    self.loans[loan_id].lease_id == lease_id
                    for loan_id in affected_loans
                )
            )
        }
        queued_event_seqs = {
            entry.event_seq
            for entry in self.queue.values()
            if entry.status == "PENDING" and entry.declaration_version == old_version
        }
        requeue_events: dict[int, int] = {}
        for lease_id in sorted(affected_lease_ids):
            lease = self.leases[lease_id]
            elapsed = max(0, at_ms - lease.grant_ms)
            remaining = max(0, lease.remaining_occupancy_ms - elapsed)
            reason = self._invalidation_reason(
                lease.function_id,
                changed_functions,
                changed_edges,
                global_changed,
                affected_loans,
                lease_id,
            )
            self._invalidate_lease(lease, at_ms, new_version, reason)
            if remaining > 0 and lease.function_id in new_declaration.functions:
                requeue_events.setdefault(lease.event_seq, remaining)

        stale_queues: list[QueueEntry] = []
        for entry in sorted(self.queue.values(), key=lambda item: item.queue_seq):
            if entry.status != "PENDING" or entry.declaration_version != old_version:
                continue
            if global_changed or entry.function_id in affected_functions:
                entry.status = "STALE"
                entry.resolved_ms = at_ms
                stale_queues.append(entry)
                self._append_invalidation(
                    kind="QUEUE",
                    object_id=entry.queue_id,
                    lease_id=entry.lease_id,
                    event_seq=entry.event_seq,
                    reason=self._queue_invalidation_reason(
                        entry.function_id, changed_functions, changed_edges, global_changed
                    ),
                    old_version=old_version,
                    new_version=new_version,
                    at_ms=at_ms,
                )
                if entry.function_id in new_declaration.functions:
                    requeue_events.setdefault(
                        entry.event_seq, entry.remaining_occupancy_ms
                    )
            else:
                entry.declaration_version = new_version

        for preemption in sorted(
            self.preemptions.values(), key=lambda item: item.preemption_id
        ):
            if preemption.declaration_version >= new_version:
                continue
            endpoint_affected = (
                global_changed
                or preemption.beneficiary_function in affected_functions
                or preemption.victim_function in affected_functions
            )
            if preemption.status == "SCHEDULED":
                preemption.status = "STALE" if endpoint_affected else "STALE_VERSION"
                victim = self.leases.get(preemption.victim_lease_id)
                if victim is not None and victim.status == ACTIVE:
                    self._restore_natural_schedule(victim)
            elif preemption.status == "SUSPENDED":
                preemption.status = "STALE_VERSION"
            else:
                continue
            self._append_invalidation(
                kind="PREEMPTION",
                object_id=preemption.preemption_id,
                lease_id=preemption.victim_lease_id,
                event_seq=preemption.beneficiary_event_seq,
                reason=(
                    "PREEMPTION_RELATION_OR_ENDPOINT_STALE"
                    if endpoint_affected
                    else "DECLARATION_VERSION_SUPERSEDED"
                ),
                old_version=preemption.declaration_version,
                new_version=new_version,
                at_ms=at_ms,
            )

        for lease in self.leases.values():
            if lease.status == ACTIVE:
                lease.current_version = new_version
        for loan in self.loans.values():
            if loan.status == "ACTIVE":
                loan.current_version = new_version

        for event_seq in sorted(requeue_events):
            existing_pending = any(
                item.status == "PENDING" and item.event_seq == event_seq
                for item in self.queue.values()
            )
            if existing_pending:
                continue
            original = self._event_for_seq(event_seq)
            self._enqueue(
                event_seq=event_seq,
                function_id=original.function_id,
                demand=original.concurrency_demand,
                remaining_occupancy_ms=requeue_events[event_seq],
                arrived_ms=original.arrival_ms,
                enqueued_ms=at_ms,
                not_before_ms=at_ms,
                source="DECLARATION_REPLACEMENT",
            )

        self.declaration = new_declaration
        self.function_used = {
            function_id: self.function_used.get(function_id, 0)
            for function_id in sorted(new_declaration.functions)
        }
        self.latest_watermarks = []
        self._op(
            "DECLARATION_APPLIED",
            old_version=old_version,
            new_version=new_version,
            at_ms=at_ms,
            changed_functions=sorted(changed_functions),
            changed_preemptions=[list(edge) for edge in sorted(changed_edges)],
            invalidated_leases=sorted(affected_lease_ids),
            stale_queues=[entry.queue_id for entry in stale_queues],
        )
        return True

    def _invalidate_lease(
        self, lease: Lease, at_ms: int, new_version: int, reason: str
    ) -> None:
        lease.status = STALE
        lease.release_ms = at_ms
        lease.release_disposition = "STALE_INVALIDATED"
        lease.release_reason = reason
        lease.invalidation_reason = reason
        lease.current_version = new_version
        self.global_used -= lease.concurrency
        self.function_used[lease.function_id] -= lease.concurrency
        for loan_id in lease.loan_ids:
            loan = self.loans.get(loan_id)
            if loan is not None and loan.status == "ACTIVE":
                loan.status = "STALE"
                loan.current_version = new_version
                loan.returned_ms = at_ms
                loan.disposition = "INVALIDATED"
                loan.reason = reason
                self._op("LOAN_RETURN", loan=loan.as_dict())
        self._append_invalidation(
            kind="LEASE",
            object_id=lease.lease_id,
            lease_id=lease.lease_id,
            event_seq=lease.event_seq,
            reason=reason,
            old_version=lease.origin_version,
            new_version=new_version,
            at_ms=at_ms,
        )
        self._op("INVALIDATE_LEASE", lease=lease.as_dict())

    def _invalidation_reason(
        self,
        function_id: str,
        changed_functions: set[str],
        changed_edges: set[tuple[str, str]],
        global_changed: bool,
        affected_loans: set[str],
        lease_id: str,
    ) -> str:
        if global_changed:
            return "GLOBAL_CONCURRENCY_CHANGED"
        if any(self.loans[loan_id].lease_id == lease_id for loan_id in affected_loans):
            return "BORROW_REFERENCE_CHANGED"
        if function_id in changed_functions:
            return "FUNCTION_PARAMETERS_CHANGED"
        if any(function_id in edge for edge in changed_edges):
            return "PREEMPTION_RELATION_CHANGED"
        return "REFERENCE_CLOSURE_CHANGED"

    @staticmethod
    def _queue_invalidation_reason(
        function_id: str,
        changed_functions: set[str],
        changed_edges: set[tuple[str, str]],
        global_changed: bool,
    ) -> str:
        if global_changed:
            return "GLOBAL_CONCURRENCY_CHANGED"
        if function_id in changed_functions:
            return "FUNCTION_PARAMETERS_CHANGED"
        if any(function_id in edge for edge in changed_edges):
            return "PREEMPTION_RELATION_CHANGED"
        return "DECLARATION_REPLACEMENT"

    def _append_invalidation(
        self,
        *,
        kind: str,
        object_id: str,
        lease_id: str | None,
        event_seq: int,
        reason: str,
        old_version: int,
        new_version: int,
        at_ms: int,
    ) -> None:
        item = {
            "kind": kind,
            "object_id": object_id,
            "lease_id": lease_id,
            "event_seq": event_seq,
            "reason": reason,
            "old_version": old_version,
            "new_version": new_version,
            "at_ms": at_ms,
        }
        if item not in self.invalidations:
            self.invalidations.append(item)

    def watermarks(self) -> list[dict[str, Any]]:
        now = self.clock_ms
        function_values: list[tuple[str, Ratio, dict[str, Any]]] = []
        for function_id in sorted(self.declaration.functions):
            function = self.declaration.functions[function_id]
            borrowed = self._borrowed_in(function_id)
            lent = self._lent_out(function_id)
            used = self.function_used.get(function_id, 0)
            capacity = max(0, function.max_concurrency + borrowed - lent)
            queue_length = sum(
                1
                for entry in self.queue.values()
                if entry.status == "PENDING" and entry.function_id == function_id
            )
            near_borrow_units = sum(
                loan.amount
                for loan in self.loans.values()
                if loan.status == "ACTIVE"
                and loan.borrower_id == function_id
                and loan.return_deadline_ms <= now + self.session.borrow_return_error_ms
            )
            ratio = quantize_ratio(
                used,
                capacity,
                self.session.watermark_scale,
                near_borrow_units=near_borrow_units,
            )
            function_values.append(
                (
                    function_id,
                    ratio,
                    {
                        "scope": "FUNCTION",
                        "function_id": function_id,
                        "used_concurrency": used,
                        "borrowed_concurrency": borrowed,
                        "lent_concurrency": lent,
                        "queue_length": queue_length,
                        "capacity": capacity,
                        "min_concurrency": function.min_concurrency,
                        "max_concurrency": function.max_concurrency,
                        "near_deadline_borrowed": near_borrow_units,
                    },
                )
            )

        ranked = rank_ratios([(key, ratio) for key, ratio, _ in function_values])
        affected = {item.key: item.affected_by_error for item in ranked}
        result: list[dict[str, Any]] = []
        for function_id, ratio, extra in function_values:
            result.append(
                {
                    **extra,
                    "ratio_fp": ratio.fixed,
                    "ratio_scale": ratio.scale,
                    "ratio_numerator": ratio.numerator,
                    "ratio_denominator": ratio.denominator,
                    "rounding": "FLOOR_TOWARD_ZERO",
                    "single_value_error_fp": ratio.single_value_error_fp,
                    "comparison_error_fp": ratio.comparison_error_fp,
                    "borrow_return_error_fp": ratio.borrow_return_error_fp,
                    "total_comparison_error_fp": ratio.total_comparison_error_fp,
                    "dominant_error_segment": ratio.dominant_error_segment,
                    "comparison_affected_by_error": affected[function_id],
                }
            )

        global_ratio = quantize_ratio(
            self.global_used,
            self.declaration.global_concurrency,
            self.session.watermark_scale,
            near_borrow_units=sum(
                loan.amount
                for loan in self.loans.values()
                if loan.status == "ACTIVE"
                and loan.return_deadline_ms <= now + self.session.borrow_return_error_ms
            ),
        )
        result.append(
            {
                "scope": "GLOBAL",
                "function_id": None,
                "used_concurrency": self.global_used,
                "borrowed_concurrency": self._borrowed_global(),
                "lent_concurrency": self._lent_global(),
                "queue_length": sum(
                    1 for entry in self.queue.values() if entry.status == "PENDING"
                ),
                "capacity": self.declaration.global_concurrency,
                "min_concurrency": 0,
                "max_concurrency": self.declaration.global_concurrency,
                "near_deadline_borrowed": sum(
                    loan.amount
                    for loan in self.loans.values()
                    if loan.status == "ACTIVE"
                    and loan.return_deadline_ms
                    <= now + self.session.borrow_return_error_ms
                ),
                "ratio_fp": global_ratio.fixed,
                "ratio_scale": global_ratio.scale,
                "ratio_numerator": global_ratio.numerator,
                "ratio_denominator": global_ratio.denominator,
                "rounding": "FLOOR_TOWARD_ZERO",
                "single_value_error_fp": global_ratio.single_value_error_fp,
                "comparison_error_fp": global_ratio.comparison_error_fp,
                "borrow_return_error_fp": global_ratio.borrow_return_error_fp,
                "total_comparison_error_fp": global_ratio.total_comparison_error_fp,
                "dominant_error_segment": global_ratio.dominant_error_segment,
                "comparison_affected_by_error": False,
            }
        )
        return result

    def snapshot(self) -> dict[str, Any]:
        active_leases = [
            lease.as_dict()
            for lease in sorted(self.leases.values(), key=lambda item: item.lease_id)
            if lease.status == ACTIVE
        ]
        pending_queue = [
            entry.as_dict()
            for entry in sorted(self.queue.values(), key=lambda item: item.queue_seq)
            if entry.status == "PENDING"
        ]
        active_loans = [
            loan.as_dict()
            for loan in sorted(self.loans.values(), key=lambda item: item.loan_id)
            if loan.status == "ACTIVE"
        ]
        preemptions = [
            item.as_dict()
            for item in sorted(
                self.preemptions.values(), key=lambda entry: entry.preemption_id
            )
            if item.declaration_version >= self.declaration.declaration_version
            and item.status not in {"STALE", "STALE_VERSION"}
        ]
        return {
            "declaration_version": self.declaration.declaration_version,
            "declaration_digest": self.declaration.digest,
            "clock_ms": self.clock_ms,
            "scale": self.session.watermark_scale,
            "global_used": self.global_used,
            "global_budget": self.declaration.global_concurrency,
            "functions": [self._function_snapshot(key) for key in sorted(self.declaration.functions)],
            "leases": active_leases,
            "loans": active_loans,
            "queue": pending_queue,
            "preemptions": preemptions,
            "watermarks": self.latest_watermarks or self.watermarks(),
            "invalidations": sorted(
                self.invalidations,
                key=lambda item: (item["new_version"], item["kind"], item["object_id"]),
            ),
        }

    def _function_snapshot(self, function_id: str) -> dict[str, Any]:
        function = self.declaration.functions[function_id]
        borrowed = self._borrowed_in(function_id)
        lent = self._lent_out(function_id)
        return {
            "function_id": function_id,
            "priority": function.priority,
            "memory_mib": function.memory_mib,
            "instance_concurrency": function.instance_concurrency,
            "lease_ms": function.lease_ms,
            "min_concurrency": function.min_concurrency,
            "max_concurrency": function.max_concurrency,
            "used_concurrency": self.function_used[function_id],
            "borrowed_concurrency": borrowed,
            "lent_concurrency": lent,
            "effective_capacity": max(0, function.max_concurrency + borrowed - lent),
            "queue_length": sum(
                1
                for entry in self.queue.values()
                if entry.status == "PENDING" and entry.function_id == function_id
            ),
        }

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "state_format_version": 1,
            "declaration": self.declaration.as_dict(include_digest=False),
            "session": self.session.as_dict(),
            "clock_ms": self.clock_ms,
            "global_used": self.global_used,
            "function_used": dict(sorted(self.function_used.items())),
            "leases": [
                lease.as_dict()
                for lease in sorted(self.leases.values(), key=lambda item: item.lease_id)
            ],
            "queue": [
                entry.as_dict()
                for entry in sorted(self.queue.values(), key=lambda item: item.queue_seq)
            ],
            "loans": [
                loan.as_dict()
                for loan in sorted(self.loans.values(), key=lambda item: item.loan_id)
            ],
            "preemptions": [
                item.as_dict()
                for item in sorted(self.preemptions.values(), key=lambda entry: entry.preemption_id)
            ],
            "releases": [record.as_dict() for record in self.releases],
            "invalidations": sorted(
                self.invalidations,
                key=lambda item: (item["new_version"], item["kind"], item["object_id"]),
            ),
            "latest_watermarks": self.latest_watermarks,
            "queue_seq": self.queue_seq,
            "grant_counts": {str(key): self.grant_counts[key] for key in sorted(self.grant_counts)},
            "events": [
                self.events_index[key].as_dict() for key in sorted(self.events_index)
            ],
        }

    @classmethod
    def from_state_dict(
        cls,
        declaration: Declaration,
        session: SessionConfig,
        raw: dict[str, Any],
    ) -> "LeaseEngine":
        engine = cls(declaration, session)
        engine.clock_ms = int(raw["clock_ms"])
        engine.global_used = int(raw["global_used"])
        engine.function_used = {
            str(key): int(value) for key, value in raw.get("function_used", {}).items()
        }
        engine.function_used = {
            key: engine.function_used.get(key, 0)
            for key in sorted(declaration.functions)
        }
        engine.leases = {
            item["lease_id"]: Lease.from_dict(dict(item))
            for item in raw.get("leases", [])
        }
        engine.queue = {
            item["queue_id"]: QueueEntry.from_dict(dict(item))
            for item in raw.get("queue", [])
        }
        engine.loans = {
            item["loan_id"]: Loan.from_dict(dict(item))
            for item in raw.get("loans", [])
        }
        engine.preemptions = {
            item["preemption_id"]: Preemption.from_dict(dict(item))
            for item in raw.get("preemptions", [])
        }
        engine.releases = [
            ReleaseRecord.from_dict(dict(item)) for item in raw.get("releases", [])
        ]
        engine.invalidations = list(raw.get("invalidations", []))
        engine.latest_watermarks = list(raw.get("latest_watermarks", []))
        engine.queue_seq = int(raw.get("queue_seq", 0))
        engine.grant_counts = {
            int(key): int(value) for key, value in raw.get("grant_counts", {}).items()
        }
        engine.events_index = {
            int(item["event_seq"]): CallEvent(**item) for item in raw.get("events", [])
        }
        engine._rebuild_schedules()
        engine._validate_accounting()
        return engine

    def _rebuild_schedules(self) -> None:
        for lease in sorted(self.leases.values(), key=lambda item: item.lease_id):
            if lease.status == ACTIVE:
                loans = [
                    self.loans[loan_id]
                    for loan_id in lease.loan_ids
                    if loan_id in self.loans and self.loans[loan_id].status == "ACTIVE"
                ]
                self._schedule_initial_release(lease, loans)
        for preemption in sorted(
            self.preemptions.values(), key=lambda item: item.preemption_id
        ):
            if preemption.status != "SCHEDULED":
                continue
            lease = self.leases.get(preemption.victim_lease_id)
            if lease is not None and lease.status == ACTIVE:
                self._schedule_lease_release(
                    lease,
                    preemption.suspend_at_ms,
                    "PREEMPTED",
                    1,
                    preemption.preemption_id,
                )

    def _validate_accounting(self) -> None:
        expected_global = sum(
            lease.concurrency
            for lease in self.leases.values()
            if lease.status == ACTIVE
        )
        expected_functions = {key: 0 for key in self.declaration.functions}
        for lease in self.leases.values():
            if lease.status == ACTIVE:
                expected_functions[lease.function_id] = (
                    expected_functions.get(lease.function_id, 0) + lease.concurrency
                )
        if expected_global != self.global_used:
            raise QuotaError(
                "ledger_state.global_used",
                "recovered global accounting mismatch",
                expected=expected_global,
                actual=self.global_used,
            )
        for function_id in sorted(expected_functions):
            if expected_functions[function_id] != self.function_used.get(function_id, 0):
                raise QuotaError(
                    "ledger_state.function_used",
                    "recovered function accounting mismatch",
                    function_id=function_id,
                    expected=expected_functions[function_id],
                    actual=self.function_used.get(function_id, 0),
                )
        if self.global_used > self.declaration.global_concurrency:
            raise QuotaError(
                "ledger_state.global_used",
                "recovered state exceeds global budget",
                used=self.global_used,
                budget=self.declaration.global_concurrency,
            )

    def _borrow_relations(self, function_id: str) -> list[Any]:
        return sorted(
            [edge for edge in self.declaration.preemptions if edge.higher == function_id],
            key=lambda edge: (
                self.declaration.functions[edge.lower].priority,
                edge.lower,
            ),
        )

    def _effective_capacity(self, function_id: str) -> int:
        function = self.declaration.functions[function_id]
        return max(
            0,
            function.max_concurrency
            + self._borrowed_in(function_id)
            - self._lent_out(function_id),
        )

    def _lender_surplus(self, function_id: str) -> int:
        function = self.declaration.functions[function_id]
        baseline = max(self.function_used.get(function_id, 0), function.min_concurrency)
        return max(
            0,
            function.max_concurrency - baseline - self._lent_out(function_id),
        )

    def _borrowed_in(self, function_id: str) -> int:
        return sum(
            loan.amount
            for loan in self.loans.values()
            if loan.status == "ACTIVE" and loan.borrower_id == function_id
        )

    def _lent_out(self, function_id: str) -> int:
        return sum(
            loan.amount
            for loan in self.loans.values()
            if loan.status == "ACTIVE" and loan.lender_id == function_id
        )

    def _borrowed_global(self) -> int:
        return sum(loan.amount for loan in self.loans.values() if loan.status == "ACTIVE")

    def _lent_global(self) -> int:
        return self._borrowed_global()

    def _op(self, kind: str, **payload: Any) -> None:
        self.current_ops.append({"type": kind, **payload})

    def _event_arrival_ms(self, event_seq: int) -> int:
        return self._event_for_seq(event_seq).arrival_ms

    def _event_for_seq(self, event_seq: int) -> CallEvent:
        event = self.events_index.get(event_seq)
        if event is None:
            raise KeyError(f"event not present in engine state: {event_seq}")
        return event


def clone_engine(engine: LeaseEngine) -> LeaseEngine:
    return LeaseEngine.from_state_dict(
        engine.declaration,
        engine.session,
        engine.to_state_dict(),
    )

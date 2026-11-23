from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .clock import Clock, RealMonotonicClock
from .errors import BusinessError, ErrorCode
from .ledger import AuditLedger, EventRecord
from .logging_utils import StructuredLogger
from .scenario import WATERMARK_TRUNCATION, Scenario, SourceSpec
from .util import canonical_digest, is_int


@dataclass
class Lease:
    lease_id: str
    source_id: str
    amount: int
    issued_at_ms: int
    expires_at_ms: int
    renew_count: int = 0
    last_renewed_at_ms: int | None = None

    def clone(self) -> "Lease":
        return Lease(
            lease_id=self.lease_id,
            source_id=self.source_id,
            amount=self.amount,
            issued_at_ms=self.issued_at_ms,
            expires_at_ms=self.expires_at_ms,
            renew_count=self.renew_count,
            last_renewed_at_ms=self.last_renewed_at_ms,
        )

    def as_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "source_id": self.source_id,
            "amount": self.amount,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "renew_count": self.renew_count,
            "last_renewed_at_ms": self.last_renewed_at_ms,
        }


@dataclass
class SourceRuntime:
    spec: SourceSpec
    retained_quota: int
    active_leases: dict[str, Lease] = field(default_factory=dict)
    reclaimed_total: int = 0
    low_streak: int = 0
    observation_count: int = 0
    last_ratio_ppm: int = 0
    last_observed_at_ms: int | None = None
    last_release_at_ms: int | None = None
    last_reclaim_seq: int | None = None
    last_reclaim_basis: str | None = None
    observation_history: list[dict] = field(default_factory=list)

    @property
    def active_granted(self) -> int:
        return sum(lease.amount for lease in self.active_leases.values())

    @property
    def reclaimable(self) -> int:
        return max(0, self.retained_quota - max(self.spec.reserve_quota, self.active_granted))

    def as_dict(self) -> dict:
        return {
            "source_id": self.spec.source_id,
            "lease_share": self.spec.lease_share,
            "reserve_quota": self.spec.reserve_quota,
            "retained_quota": self.retained_quota,
            "active_granted": self.active_granted,
            "reclaimed_total": self.reclaimed_total,
            "low_streak": self.low_streak,
            "observation_count": self.observation_count,
            "last_ratio_ppm": self.last_ratio_ppm,
            "last_observed_at_ms": self.last_observed_at_ms,
            "last_release_at_ms": self.last_release_at_ms,
            "last_reclaim_seq": self.last_reclaim_seq,
            "last_reclaim_basis": self.last_reclaim_basis,
            "active_leases": [lease.as_dict() for lease in self._ordered_leases()],
        }

    def _ordered_leases(self) -> list[Lease]:
        return [self.active_leases[key] for key in sorted(self.active_leases)]


class LeaseEngine:
    """Monotonic-clock lease engine backed by an append-only audit ledger."""

    def __init__(
        self,
        scenario: Scenario,
        ledger: AuditLedger,
        *,
        clock: Clock | None = None,
        logger: StructuredLogger | None = None,
        crash_after_intent: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.scenario = scenario
        self.ledger = ledger
        self.clock = clock or RealMonotonicClock()
        self.logger = logger
        self.crash_after_intent = crash_after_intent
        self._lock = threading.RLock()
        self.sources: dict[str, SourceRuntime] = {
            spec.source_id: SourceRuntime(spec=spec, retained_quota=spec.reserve_quota)
            for spec in scenario.ordered_sources()
        }
        self.global_allocatable = scenario.initial_global_allocatable
        self._last_event_mono_ms: int | None = None
        self._retired_leases: set[str] = set()
        self._retired_lease_sources: dict[str, str] = {}
        self._next_observation_ms: int | None = None
        self._unknowns: list[dict] = []
        self._recover_committed_events()
        self._resolve_pending_intents()
        self._assert_invariants()

    def _recover_committed_events(self) -> None:
        for event in self.ledger.events():
            if self._last_event_mono_ms is not None and event.mono_ms < self._last_event_mono_ms:
                raise BusinessError(
                    ErrorCode.E_CLOCK,
                    "ledger contains a backwards monotonic timestamp",
                    field_path="mono_ms",
                    details={"previous": self._last_event_mono_ms, "actual": event.mono_ms},
                    event_seq=event.seq,
                )
            self._last_event_mono_ms = event.mono_ms
            if event.event_type == "LEASE_ISSUED":
                source = self._source_or_ledger_error(event.source_id, event.seq)
                lease = Lease(
                    lease_id=str(event.lease_id),
                    source_id=source.spec.source_id,
                    amount=int(event.amount or 0),
                    issued_at_ms=int(event.payload["issued_at_ms"]),
                    expires_at_ms=int(event.payload["expires_at_ms"]),
                )
                previous_retained = source.retained_quota
                source.active_leases[lease.lease_id] = lease
                needed = max(0, source.active_granted - previous_retained)
                source.retained_quota += needed
                self.global_allocatable -= needed
                source.low_streak = 0
            elif event.event_type == "LEASE_RENEWED":
                source = self._source_or_ledger_error(event.source_id, event.seq)
                lease = source.active_leases.get(str(event.lease_id))
                if lease is None:
                    raise BusinessError(
                        ErrorCode.E_LEDGER,
                        "renewal references no active lease during recovery",
                        details={"lease_id": event.lease_id},
                        event_seq=event.seq,
                        source_id=source.spec.source_id,
                    )
                lease.expires_at_ms = int(event.payload["new_expires_at_ms"])
                lease.renew_count = int(event.payload.get("renew_count", lease.renew_count + 1))
                lease.last_renewed_at_ms = event.mono_ms
            elif event.event_type in {"LEASE_EXPIRED", "LEASE_RELEASED"}:
                source = self._source_or_ledger_error(event.source_id, event.seq)
                lease = source.active_leases.pop(str(event.lease_id), None)
                if lease is None:
                    raise BusinessError(
                        ErrorCode.E_LEDGER,
                        "lease release references no active lease during recovery",
                        details={"lease_id": event.lease_id},
                        event_seq=event.seq,
                        source_id=source.spec.source_id,
                    )
                self._retired_leases.add(lease.lease_id)
                self._retired_lease_sources[lease.lease_id] = lease.source_id
                source.low_streak = 0
                source.last_release_at_ms = event.mono_ms
            elif event.event_type == "QUOTA_RECLAIMED":
                source = self._source_or_ledger_error(event.source_id, event.seq)
                amount = int(event.amount or 0)
                if amount < 0 or source.retained_quota < amount:
                    raise BusinessError(
                        ErrorCode.E_LEDGER,
                        "quota reclaim amount is invalid during recovery",
                        event_seq=event.seq,
                        source_id=source.spec.source_id,
                    )
                source.retained_quota -= amount
                self.global_allocatable += amount
                source.reclaimed_total += amount
                source.last_reclaim_seq = event.seq
                source.last_reclaim_basis = event.payload.get("basis")
            elif event.event_type == "OBSERVED":
                source = self._source_or_ledger_error(event.source_id, event.seq)
                observation = event.payload.get("observation", event.payload)
                source.low_streak = int(observation["low_streak_after"])
                source.observation_count = int(observation["observation_index"])
                source.last_ratio_ppm = int(observation["watermark_ratio_ppm"])
                source.last_observed_at_ms = event.mono_ms
                source.observation_history.append(
                    {
                        "observation_index": int(observation["observation_index"]),
                        "at_ms": event.mono_ms,
                        "active_granted": int(observation["active_granted"]),
                        "lease_share": int(observation["lease_share"]),
                        "watermark_ratio_ppm": source.last_ratio_ppm,
                        "below_threshold": bool(observation["below_threshold"]),
                        "low_streak_after": source.low_streak,
                    }
                )
                source.observation_history = source.observation_history[
                    -(self.scenario.reclaim_window_rounds + 1) :
                ]
                self._next_observation_ms = event.mono_ms + self.scenario.observation_window_ms
            elif event.event_type in {"RACE_REJECTED", "LEASE_UNKNOWN"}:
                pass
            else:
                raise BusinessError(
                    ErrorCode.E_LEDGER,
                    "unknown event type during recovery",
                    field_path="event_type",
                    details={"event_type": event.event_type},
                    event_seq=event.seq,
                )
            self._assert_invariants(event_seq=event.seq)

    def _resolve_pending_intents(self) -> None:
        pending = self.ledger.pending_intents()
        if not pending:
            return
        now = max(self.clock.now_ms(), self._last_event_mono_ms or 0)
        for intent in pending:
            source = self._source_or_ledger_error(str(intent["source_id"]), None)
            payload = self._payload(
                decision="MARK_UNKNOWN",
                basis="RECOVERY:committed_intent_without_committed_lease_event",
                source=source,
                extra={
                    "attempted_amount": int(intent["amount"]),
                    "intent_id": intent["intent_id"],
                    "unknown": True,
                    "exit_report_available": False,
                },
            )
            event = self.ledger.resolve_unknown_intent(intent, mono_ms=now, payload=payload)
            self._last_event_mono_ms = now
            self._unknowns.append(
                {
                    "intent_id": intent["intent_id"],
                    "source_id": intent["source_id"],
                    "amount": intent["amount"],
                    "mono_ms": intent["mono_ms"],
                    "event_seq": event.seq,
                    "basis": "RECOVERY:committed_intent_without_committed_lease_event",
                }
            )
            if self.logger is not None:
                self.logger.info(
                    "E_RECOVERY_UNKNOWN",
                    "uncommitted issuance intent marked unknown; reissue required",
                    intent_id=intent["intent_id"],
                    source_id=intent["source_id"],
                    event_seq=event.seq,
                )

    def _source_or_ledger_error(self, source_id: str | None, seq: int | None) -> SourceRuntime:
        if source_id is None or source_id not in self.sources:
            raise BusinessError(
                ErrorCode.E_LEDGER,
                "event references an unknown source",
                field_path="source_id",
                details={"source_id": source_id},
                event_seq=seq,
                source_id=source_id,
            )
        return self.sources[source_id]

    def _now(self) -> int:
        now = self.clock.now_ms()
        if self._last_event_mono_ms is not None and now < self._last_event_mono_ms:
            raise BusinessError(
                ErrorCode.E_CLOCK,
                "monotonic clock moved backwards relative to the ledger",
                field_path="mono_ms",
                details={"ledger_mono_ms": self._last_event_mono_ms, "clock_mono_ms": now},
            )
        return now

    def _coerce_now(self, now_ms: int | None) -> int:
        now = self._now() if now_ms is None else int(now_ms)
        if self._last_event_mono_ms is not None and now < self._last_event_mono_ms:
            return self._last_event_mono_ms
        return now

    def _ownership(self) -> list[dict]:
        records: list[dict] = []
        for source_id in sorted(self.sources):
            source = self.sources[source_id]
            for lease in source._ordered_leases():
                records.append(
                    {
                        "lease_id": lease.lease_id,
                        "source_id": lease.source_id,
                        "amount": lease.amount,
                        "issued_at_ms": lease.issued_at_ms,
                        "expires_at_ms": lease.expires_at_ms,
                        "renew_count": lease.renew_count,
                        "last_renewed_at_ms": lease.last_renewed_at_ms,
                    }
                )
        return records

    def _payload(
        self,
        *,
        decision: str,
        basis: str,
        source: SourceRuntime | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict:
        ownership = self._ownership()
        payload: dict[str, Any] = {
            "decision": decision,
            "basis": basis,
            "global_allocatable_after": self.global_allocatable,
            "total_active_after": self.total_active,
            "retained_total_after": self.retained_total,
            "ownership": ownership,
            "ownership_digest": canonical_digest(ownership),
        }
        if source is not None:
            payload["source_state_after"] = source.as_dict()
        if extra:
            payload.update(extra)
        return payload

    def _commit(
        self,
        *,
        event_type: str,
        mono_ms: int,
        source_id: str | None,
        lease_id: str | None,
        amount: int | None,
        payload: Mapping[str, Any],
        intent_id: str | None = None,
    ) -> EventRecord:
        event = self.ledger.append_event(
            event_type=event_type,
            mono_ms=mono_ms,
            source_id=source_id,
            lease_id=lease_id,
            amount=amount,
            payload=payload,
            intent_id=intent_id,
        )
        self._last_event_mono_ms = mono_ms
        self._assert_invariants(event_seq=event.seq)
        if self.logger is not None:
            self.logger.info(
                "AUDIT_EVENT",
                event_type,
                seq=event.seq,
                source_id=source_id,
                lease_id=lease_id,
                amount=amount,
                decision=payload.get("decision"),
            )
        return event

    def _race(
        self,
        *,
        source_id: str | None,
        lease_id: str | None,
        mono_ms: int,
        attempted_operation: str,
        basis: str,
        message: str,
    ) -> None:
        source = self.sources.get(source_id) if source_id is not None else None
        payload = self._payload(
            decision="REJECT_RACE",
            basis=basis,
            source=source,
            extra={
                "attempted_operation": attempted_operation,
                "attempted_lease_id": lease_id,
                "code": "E_RACE",
                "ownership_unchanged": True,
            },
        )
        event = self._commit(
            event_type="RACE_REJECTED",
            mono_ms=mono_ms,
            source_id=source_id,
            lease_id=lease_id,
            amount=None,
            payload=payload,
        )
        raise BusinessError(
            ErrorCode.E_RACE,
            message,
            field_path="lease_id" if lease_id else "source_id",
            details={"attempted_operation": attempted_operation, "basis": basis},
            event_seq=event.seq,
            source_id=source_id,
        )

    def process_expirations(self, now_ms: int | None = None) -> list[EventRecord]:
        with self._lock:
            now = self._coerce_now(now_ms)
            if self._last_event_mono_ms is not None and now < self._last_event_mono_ms:
                now = self._last_event_mono_ms
            return self._process_expirations_locked(now)

    def _process_expirations_locked(self, now: int) -> list[EventRecord]:
        expired: list[Lease] = []
        for source in self.sources.values():
            for lease in source.active_leases.values():
                if lease.expires_at_ms <= now:
                    expired.append(lease)
        expired.sort(key=lambda lease: (lease.expires_at_ms, lease.lease_id))
        events: list[EventRecord] = []
        for lease in expired:
            source = self.sources[lease.source_id]
            source.active_leases.pop(lease.lease_id, None)
            source.low_streak = 0
            source.last_release_at_ms = now
            self._retired_leases.add(lease.lease_id)
            self._retired_lease_sources[lease.lease_id] = lease.source_id
            payload = self._payload(
                decision="EXPIRE",
                basis="EXPIRY:observed_mono_ms>=expires_at_ms",
                source=source,
                extra={
                    "lease": lease.as_dict(),
                    "issued_at_ms": lease.issued_at_ms,
                    "expires_at_ms": lease.expires_at_ms,
                    "observed_mono_ms": now,
                    "released_amount": lease.amount,
                    "ownership_change": "ACTIVE_TO_IDLE_RETAINED_QUOTA",
                },
            )
            events.append(
                self._commit(
                    event_type="LEASE_EXPIRED",
                    mono_ms=now,
                    source_id=lease.source_id,
                    lease_id=lease.lease_id,
                    amount=lease.amount,
                    payload=payload,
                )
            )
        return events

    def issue(self, source_id: str, amount: int, *, now_ms: int | None = None) -> Lease:
        with self._lock:
            now = self._coerce_now(now_ms)
            self._process_expirations_locked(now)
            if not is_int(amount) or amount <= 0:
                raise BusinessError(ErrorCode.E_CAPACITY, "lease amount must be a positive integer", "amount")
            source = self.sources.get(source_id)
            if source is None:
                raise BusinessError(
                    ErrorCode.E_LEASE,
                    "unknown source",
                    field_path="source_id",
                    source_id=source_id,
                )
            if source.active_granted + amount > source.spec.lease_share:
                raise BusinessError(
                    ErrorCode.E_CAPACITY,
                    "source lease share would be exceeded",
                    field_path=f"sources[{source_id}].lease_share",
                    source_id=source_id,
                    details={
                        "active_granted": source.active_granted,
                        "new_amount": amount,
                        "lease_share": source.spec.lease_share,
                    },
                )
            needed_from_global = max(0, source.active_granted + amount - source.retained_quota)
            if needed_from_global > self.global_allocatable:
                raise BusinessError(
                    ErrorCode.E_CAPACITY,
                    "global allocatable quota is insufficient",
                    field_path="global_allocatable",
                    source_id=source_id,
                    details={
                        "needed_from_global": needed_from_global,
                        "global_allocatable": self.global_allocatable,
                        "total_active": self.total_active,
                        "cluster_capacity": self.scenario.cluster_capacity,
                    },
                )
            lease_id = f"lease-{source_id}-{self.ledger.next_seq():08d}"
            intent_id = self.ledger.begin_intent(source_id=source_id, amount=amount, mono_ms=now)
            if self.crash_after_intent is not None:
                self.crash_after_intent(
                    {
                        "intent_id": intent_id,
                        "source_id": source_id,
                        "amount": amount,
                        "mono_ms": now,
                    }
                )
            lease = Lease(
                lease_id=lease_id,
                source_id=source_id,
                amount=amount,
                issued_at_ms=now,
                expires_at_ms=now + self.scenario.lease_duration_ms,
            )
            old_retained = source.retained_quota
            old_global = self.global_allocatable
            source.retained_quota += needed_from_global
            self.global_allocatable -= needed_from_global
            source.active_leases[lease.lease_id] = lease
            source.low_streak = 0
            payload = self._payload(
                decision="GRANT",
                basis="GRANT:active+grant<=lease_share;quota_from_global<=global_allocatable",
                source=source,
                extra={
                    "lease": lease.as_dict(),
                    "issued_at_ms": lease.issued_at_ms,
                    "expires_at_ms": lease.expires_at_ms,
                    "quota_from_global": needed_from_global,
                    "retained_quota_before": old_retained,
                    "global_allocatable_before": old_global,
                    "ownership_change": "GLOBAL_OR_IDLE_QUOTA_TO_ACTIVE_LEASE",
                },
            )
            try:
                self._commit(
                    event_type="LEASE_ISSUED",
                    mono_ms=now,
                    source_id=source_id,
                    lease_id=lease.lease_id,
                    amount=amount,
                    payload=payload,
                    intent_id=intent_id,
                )
            except BaseException:
                source.active_leases.pop(lease.lease_id, None)
                source.retained_quota = old_retained
                self.global_allocatable = old_global
                raise
            return lease.clone()

    def issue_many(
        self,
        requests: Mapping[str, int] | Iterable[tuple[str, int]],
        *,
        now_ms: int | None = None,
    ) -> list[Lease]:
        """Issue same-tick requests in canonical source order.

        This is the contention-safe entry point for schedulers: callers may supply
        requests in any order, while event ordering and ownership decisions are
        always deterministic by source_id.
        """

        items = list(requests.items()) if isinstance(requests, Mapping) else list(requests)
        ordered = sorted(items, key=lambda item: (str(item[0]), int(item[1])))
        leases: list[Lease] = []
        for source_id, amount in ordered:
            leases.append(self.issue(str(source_id), int(amount), now_ms=now_ms))
        return leases

    def renew(
        self,
        lease_id: str,
        source_id: str,
        *,
        now_ms: int | None = None,
    ) -> Lease:
        with self._lock:
            now = self._coerce_now(now_ms)
            self._process_expirations_locked(now)
            source = self.sources.get(source_id)
            lease = None
            for candidate in self.sources.values():
                if lease_id in candidate.active_leases:
                    lease = candidate.active_leases[lease_id]
                    break
            if lease is None:
                basis = (
                    "RACE:renew_after_expiry_or_release"
                    if lease_id in self._retired_leases
                    else "RACE:unknown_or_inactive_lease"
                )
                self._race(
                    source_id=source_id,
                    lease_id=lease_id,
                    mono_ms=now,
                    attempted_operation="RENEW",
                    basis=basis,
                    message="renewal does not extend a non-active or expired lease",
                )
            assert lease is not None
            if source is None or lease.source_id != source_id:
                self._race(
                    source_id=source_id,
                    lease_id=lease_id,
                    mono_ms=now,
                    attempted_operation="RENEW",
                    basis=f"RACE:holder_mismatch;actual_holder={lease.source_id}",
                    message="only the current lease holder may renew",
                )
            if now >= lease.expires_at_ms:
                self._race(
                    source_id=source_id,
                    lease_id=lease_id,
                    mono_ms=now,
                    attempted_operation="RENEW",
                    basis="RACE:mono_ms>=expires_at_ms",
                    message="renewal must occur strictly before expiry",
                )
            old_expires = lease.expires_at_ms
            old_renew_count = lease.renew_count
            lease.expires_at_ms = now + self.scenario.lease_duration_ms
            lease.renew_count += 1
            lease.last_renewed_at_ms = now
            payload = self._payload(
                decision="RENEW",
                basis="RENEW:current_holder_match;mono_ms<expires_at_ms",
                source=source,
                extra={
                    "lease": lease.as_dict(),
                    "old_expires_at_ms": old_expires,
                    "new_expires_at_ms": lease.expires_at_ms,
                    "renew_count": lease.renew_count,
                    "ownership_change": "NONE_EXTEND_EXPIRY",
                },
            )
            try:
                self._commit(
                    event_type="LEASE_RENEWED",
                    mono_ms=now,
                    source_id=source_id,
                    lease_id=lease_id,
                    amount=lease.amount,
                    payload=payload,
                )
            except BaseException:
                lease.expires_at_ms = old_expires
                lease.renew_count = old_renew_count
                raise
            return lease.clone()

    def release(self, lease_id: str, *, now_ms: int | None = None) -> Lease:
        with self._lock:
            now = self._coerce_now(now_ms)
            self._process_expirations_locked(now)
            lease: Lease | None = None
            for source in self.sources.values():
                lease = source.active_leases.get(lease_id)
                if lease is not None:
                    break
            if lease is None:
                known = lease_id in self._retired_leases
                retired_source = self._retired_lease_sources.get(lease_id)
                self._race(
                    source_id=retired_source,
                    lease_id=lease_id,
                    mono_ms=now,
                    attempted_operation="RELEASE",
                    basis="RACE:duplicate_or_repeated_release" if known else "RACE:unknown_lease",
                    message="lease release is not repeatable",
                )
            assert lease is not None
            source = self.sources[lease.source_id]
            released = lease.clone()
            source.active_leases.pop(lease_id, None)
            self._retired_leases.add(lease_id)
            self._retired_lease_sources[lease_id] = lease.source_id
            source.low_streak = 0
            source.last_release_at_ms = now
            payload = self._payload(
                decision="RELEASE",
                basis="RELEASE:explicit_holder_release",
                source=source,
                extra={
                    "lease": released.as_dict(),
                    "released_amount": released.amount,
                    "ownership_change": "ACTIVE_TO_IDLE_RETAINED_QUOTA",
                },
            )
            try:
                self._commit(
                    event_type="LEASE_RELEASED",
                    mono_ms=now,
                    source_id=lease.source_id,
                    lease_id=lease_id,
                    amount=released.amount,
                    payload=payload,
                )
            except BaseException:
                source.active_leases[lease_id] = released
                self._retired_leases.discard(lease_id)
                self._retired_lease_sources.pop(lease_id, None)
                raise
            return released

    def observe_all(
        self,
        *,
        now_ms: int | None = None,
        force: bool = False,
    ) -> list[EventRecord]:
        with self._lock:
            now = self._coerce_now(now_ms)
            self._process_expirations_locked(now)
            if not force and self._next_observation_ms is not None and now < self._next_observation_ms:
                return []
            events: list[EventRecord] = []
            for source_id in sorted(self.sources):
                source = self.sources[source_id]
                active = source.active_granted
                share = source.spec.lease_share
                ratio = (active * self.scenario.watermark_scale) // share if share > 0 else 0
                below = ratio < self.scenario.reclaim_threshold
                old_streak = source.low_streak
                new_streak = old_streak + 1 if below else 0
                old_ratio = source.last_ratio_ppm
                old_count = source.observation_count
                old_observed = source.last_observed_at_ms
                source.low_streak = new_streak
                source.observation_count += 1
                source.last_ratio_ppm = ratio
                source.last_observed_at_ms = now
                observation = {
                    "observation_index": source.observation_count,
                    "at_ms": now,
                    "active_granted": active,
                    "lease_share": share,
                    "watermark_ratio_ppm": ratio,
                    "below_threshold": below,
                    "low_streak_after": new_streak,
                }
                source.observation_history.append(observation)
                source.observation_history = source.observation_history[
                    -(self.scenario.reclaim_window_rounds + 1) :
                ]
                payload = self._payload(
                    decision="OBSERVE",
                    basis="FIXED_POINT:floor(active*scale/lease_share);trunc=FLOOR_TOWARD_ZERO",
                    source=source,
                    extra={
                        "observation": observation,
                        "observation_window_ms": self.scenario.observation_window_ms,
                        "watermark_scale": self.scenario.watermark_scale,
                        "watermark_truncation": WATERMARK_TRUNCATION,
                        "reclaim_threshold": self.scenario.reclaim_threshold,
                        "reclaim_window_rounds": self.scenario.reclaim_window_rounds,
                    },
                )
                try:
                    event = self._commit(
                        event_type="OBSERVED",
                        mono_ms=now,
                        source_id=source_id,
                        lease_id=None,
                        amount=active,
                        payload=payload,
                    )
                except BaseException:
                    source.low_streak = old_streak
                    source.observation_count = old_count
                    source.last_ratio_ppm = old_ratio
                    source.last_observed_at_ms = old_observed
                    if source.observation_history:
                        source.observation_history.pop()
                    raise
                events.append(event)
            self._next_observation_ms = now + self.scenario.observation_window_ms
            for source_id in sorted(self.sources):
                source = self.sources[source_id]
                if source.low_streak > self.scenario.reclaim_window_rounds and source.reclaimable > 0:
                    events.append(self._reclaim_locked(source, now, explicit=False))
            return events

    def maybe_observe(self, *, now_ms: int | None = None) -> list[EventRecord]:
        return self.observe_all(now_ms=now_ms, force=False)

    def request_quota_reclaim(self, source_id: str, *, now_ms: int | None = None) -> EventRecord:
        with self._lock:
            now = self._coerce_now(now_ms)
            self._process_expirations_locked(now)
            source = self.sources.get(source_id)
            if source is None:
                raise BusinessError(
                    ErrorCode.E_LEASE,
                    "unknown source",
                    field_path="source_id",
                    source_id=source_id,
                )
            if source.low_streak <= self.scenario.reclaim_window_rounds or source.reclaimable <= 0:
                self._race(
                    source_id=source_id,
                    lease_id=None,
                    mono_ms=now,
                    attempted_operation="QUOTA_RECLAIM",
                    basis="RACE:no_reclaimable_idle_quota_or_streak_not_exceeded",
                    message="quota reclaim is not repeatable when no eligible idle quota remains",
                )
            return self._reclaim_locked(source, now, explicit=True)

    def _reclaim_locked(self, source: SourceRuntime, now: int, *, explicit: bool) -> EventRecord:
        amount = source.reclaimable
        if amount <= 0:
            raise BusinessError(
                ErrorCode.E_QUOTA,
                "no reclaimable idle quota",
                field_path=f"sources[{source.spec.source_id}].retained_quota",
                source_id=source.spec.source_id,
            )
        old_retained = source.retained_quota
        old_global = self.global_allocatable
        source.retained_quota -= amount
        self.global_allocatable += amount
        source.reclaimed_total += amount
        source.last_reclaim_basis = (
            "LOW_WATER:ratio_ppm<threshold;"
            f"consecutive_rounds>{self.scenario.reclaim_window_rounds};"
            "reclaim_only_idle_above_reserve"
        )
        window = list(source.observation_history)
        payload = self._payload(
            decision="RECLAIM_QUOTA",
            basis=source.last_reclaim_basis,
            source=source,
            extra={
                "reclaimed_amount": amount,
                "retained_quota_before": old_retained,
                "retained_quota_after": source.retained_quota,
                "global_allocatable_before": old_global,
                "global_allocatable_after": self.global_allocatable,
                "reserve_quota": source.spec.reserve_quota,
                "active_granted": source.active_granted,
                "reclaimable_idle_above_reserve": amount,
                "watermark_ratio_ppm": source.last_ratio_ppm,
                "reclaim_threshold": self.scenario.reclaim_threshold,
                "reclaim_window_rounds": self.scenario.reclaim_window_rounds,
                "low_streak": source.low_streak,
                "observation_window": window,
                "watermark_scale": self.scenario.watermark_scale,
                "watermark_truncation": WATERMARK_TRUNCATION,
                "ownership_change": "IDLE_RETAINED_QUOTA_TO_GLOBAL_ALLOCATABLE",
            },
        )
        try:
            event = self._commit(
                event_type="QUOTA_RECLAIMED",
                mono_ms=now,
                source_id=source.spec.source_id,
                lease_id=None,
                amount=amount,
                payload=payload,
            )
        except BaseException:
            source.retained_quota = old_retained
            self.global_allocatable = old_global
            raise
        source.last_reclaim_seq = event.seq
        return event

    @property
    def total_active(self) -> int:
        return sum(source.active_granted for source in self.sources.values())

    @property
    def retained_total(self) -> int:
        return sum(source.retained_quota for source in self.sources.values())

    @property
    def unknown_leases(self) -> list[dict]:
        return list(self._unknowns) + [
            dict(row)
            for row in self.ledger.unknown_leases()
            if row["intent_id"] not in {unknown["intent_id"] for unknown in self._unknowns}
        ]

    def _assert_invariants(self, event_seq: int | None = None) -> None:
        if self.global_allocatable < 0:
            raise BusinessError(
                ErrorCode.E_LEDGER,
                "global allocatable quota became negative",
                field_path="global_allocatable",
                details={"global_allocatable": self.global_allocatable},
                event_seq=event_seq,
            )
        if self.retained_total + self.global_allocatable != self.scenario.cluster_capacity:
            raise BusinessError(
                ErrorCode.E_LEDGER,
                "retained quota plus global allocatable must equal cluster capacity",
                field_path="global_allocatable",
                details={
                    "retained_total": self.retained_total,
                    "global_allocatable": self.global_allocatable,
                    "cluster_capacity": self.scenario.cluster_capacity,
                },
                event_seq=event_seq,
            )
        if self.total_active > self.scenario.cluster_capacity:
            raise BusinessError(
                ErrorCode.E_LEDGER,
                "total active grants exceed cluster capacity",
                field_path="total_active",
                details={
                    "total_active": self.total_active,
                    "cluster_capacity": self.scenario.cluster_capacity,
                },
                event_seq=event_seq,
            )
        for source in self.sources.values():
            if source.active_granted > source.spec.lease_share:
                raise BusinessError(
                    ErrorCode.E_LEDGER,
                    "source active grants exceed its lease share",
                    field_path=f"sources[{source.spec.source_id}].lease_share",
                    source_id=source.spec.source_id,
                    event_seq=event_seq,
                )
            if source.active_granted > source.retained_quota:
                raise BusinessError(
                    ErrorCode.E_LEDGER,
                    "source active grants exceed retained quota",
                    field_path=f"sources[{source.spec.source_id}].retained_quota",
                    source_id=source.spec.source_id,
                    event_seq=event_seq,
                )
            if source.retained_quota < source.spec.reserve_quota:
                raise BusinessError(
                    ErrorCode.E_LEDGER,
                    "retained quota fell below the reserved floor",
                    field_path=f"sources[{source.spec.source_id}].reserve_quota",
                    source_id=source.spec.source_id,
                    event_seq=event_seq,
                )

    def snapshot(self) -> dict:
        with self._lock:
            self._assert_invariants()
            return {
                "scenario_version": self.scenario.scenario_version,
                "lease_params_version": self.scenario.lease_params_version,
                "lease_params_digest": self.scenario.params_digest,
                "cluster_capacity": self.scenario.cluster_capacity,
                "global_allocatable": self.global_allocatable,
                "total_active": self.total_active,
                "retained_total": self.retained_total,
                "sources": [self.sources[key].as_dict() for key in sorted(self.sources)],
                "unknown_leases": self.unknown_leases,
                "exit_report_available": False if self.unknown_leases else True,
            }

    def close(self) -> None:
        self.ledger.close()

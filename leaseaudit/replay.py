from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import BusinessError, ErrorCode
from .ledger import AuditLedger, EventRecord
from .scenario import WATERMARK_TRUNCATION, Scenario, scenario_diff
from .util import canonical_digest, first_diff


@dataclass
class ReplayLease:
    lease_id: str
    source_id: str
    amount: int
    issued_at_ms: int
    expires_at_ms: int
    renew_count: int = 0
    last_renewed_at_ms: int | None = None

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
class ReplaySource:
    source_id: str
    lease_share: int
    reserve_quota: int
    retained_quota: int
    active: dict[str, ReplayLease] = field(default_factory=dict)
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
        return sum(lease.amount for lease in self.active.values())

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "lease_share": self.lease_share,
            "reserve_quota": self.reserve_quota,
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
            "active_leases": [
                self.active[key].as_dict() for key in sorted(self.active)
            ],
        }


class ReplayState:
    def __init__(self, params: Mapping[str, Any]) -> None:
        self.capacity = int(params["cluster_capacity"])
        self.lease_duration_ms = int(params["lease_duration_ms"])
        self.reclaim_threshold = int(params["reclaim_threshold"])
        self.reclaim_window_rounds = int(params["reclaim_window_rounds"])
        self.observation_window_ms = int(params["observation_window_ms"])
        self.watermark_scale = int(params["watermark_scale"])
        self.sources: dict[str, ReplaySource] = {}
        for item in sorted(params["sources"], key=lambda source: source["source_id"]):
            source_id = str(item["source_id"])
            reserve = int(item["reserve_quota"])
            self.sources[source_id] = ReplaySource(
                source_id=source_id,
                lease_share=int(item["lease_share"]),
                reserve_quota=reserve,
                retained_quota=reserve,
            )
        self.global_allocatable = self.capacity - sum(
            source.reserve_quota for source in self.sources.values()
        )
        self.last_mono_ms: int | None = None

    @property
    def total_active(self) -> int:
        return sum(source.active_granted for source in self.sources.values())

    @property
    def retained_total(self) -> int:
        return sum(source.retained_quota for source in self.sources.values())

    def ownership(self) -> list[dict]:
        result: list[dict] = []
        for source_id in sorted(self.sources):
            source = self.sources[source_id]
            for lease_id in sorted(source.active):
                result.append(source.active[lease_id].as_dict())
        return result

    def final_state(self) -> dict:
        return {
            "cluster_capacity": self.capacity,
            "global_allocatable": self.global_allocatable,
            "total_active": self.total_active,
            "retained_total": self.retained_total,
            "sources": [self.sources[key].as_dict() for key in sorted(self.sources)],
            "ownership_digest": canonical_digest(self.ownership()),
        }


class ReplayMismatch(BusinessError):
    pass


def _mismatch(
    seq: int,
    source_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    basis: str,
) -> None:
    raise ReplayMismatch(
        ErrorCode.E_REPLAY,
        "replay differs from the committed audit ledger",
        field_path=field,
        event_seq=seq,
        source_id=source_id,
        details={
            "expected": expected,
            "actual": actual,
            "basis": basis,
            "classification": "LEDGER_DEFECT",
        },
    )


def _expect(
    condition: bool,
    seq: int,
    source_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    basis: str,
) -> None:
    if not condition:
        _mismatch(seq, source_id, field, expected, actual, basis)


def _expect_payload_equal(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
    seq: int,
    source_id: str | None,
    basis: str,
) -> None:
    if key not in payload:
        _mismatch(seq, source_id, key, expected, "<missing>", basis)
    actual = payload[key]
    diff = first_diff(actual, expected, key)
    if diff is not None:
        field, recomputed, ledger_value = diff
        _mismatch(seq, source_id, field, recomputed, ledger_value, basis)


def _source_by_id(
    state: ReplayState,
    source_id: str | None,
    seq: int,
) -> ReplaySource:
    if source_id is None or source_id not in state.sources:
        _mismatch(seq, source_id, "source_id", "declared source", source_id, "unknown source")
    return state.sources[str(source_id)]


def _find_lease(state: ReplayState, lease_id: str | None, seq: int) -> tuple[ReplaySource, ReplayLease]:
    if lease_id is None:
        _mismatch(seq, None, "lease_id", "non-null lease id", lease_id, "lease id is missing")
    for source in state.sources.values():
        if lease_id in source.active:
            return source, source.active[str(lease_id)]
    _mismatch(seq, None, "lease_id", "active lease", lease_id, "lease is not active")
    raise AssertionError("unreachable")


def _derive_decision(event: EventRecord) -> tuple[str, str | None]:
    event_type = event.event_type
    if event_type == "LEASE_ISSUED":
        return "GRANT", "GRANT:active+grant<=lease_share;quota_from_global<=global_allocatable"
    if event_type == "LEASE_RENEWED":
        return "RENEW", "RENEW:current_holder_match;mono_ms<expires_at_ms"
    if event_type == "LEASE_EXPIRED":
        return "EXPIRE", "EXPIRY:observed_mono_ms>=expires_at_ms"
    if event_type == "LEASE_RELEASED":
        return "RELEASE", "RELEASE:explicit_holder_release"
    if event_type == "QUOTA_RECLAIMED":
        return "RECLAIM_QUOTA", None
    if event_type == "OBSERVED":
        return "OBSERVE", None
    if event_type == "RACE_REJECTED":
        return "REJECT_RACE", None
    if event_type == "LEASE_UNKNOWN":
        return "MARK_UNKNOWN", "RECOVERY:committed_intent_without_committed_lease_event"
    return "", None


def _apply_event(state: ReplayState, event: EventRecord) -> None:
    event_type = event.event_type
    seq = event.seq
    source_id = event.source_id
    decision, fixed_basis = _derive_decision(event)
    if not decision:
        _mismatch(seq, source_id, "event_type", "known event type", event_type, "unknown event type")
    _expect_payload_equal(event.payload, "decision", decision, seq, source_id, "decision comparison")
    if fixed_basis is not None:
        _expect_payload_equal(event.payload, "basis", fixed_basis, seq, source_id, "decision basis comparison")
    elif event_type in {"OBSERVED", "RACE_REJECTED", "QUOTA_RECLAIMED"}:
        basis = event.payload.get("basis")
        _expect(
            isinstance(basis, str) and bool(basis),
            seq,
            source_id,
            "basis",
            "non-empty decision basis",
            basis,
            "decision basis comparison",
        )

    if event_type == "LEASE_ISSUED":
        source = _source_by_id(state, source_id, seq)
        amount = event.amount
        _expect(
            isinstance(amount, int) and not isinstance(amount, bool) and amount > 0,
            seq,
            source_id,
            "amount",
            "positive integer",
            amount,
            "issue amount validation",
        )
        assert amount is not None
        _expect(
            source.active_granted + amount <= source.lease_share,
            seq,
            source_id,
            "source_state_after.active_granted",
            f"<= {source.lease_share}",
            source.active_granted + amount,
            "same-source lease share cap",
        )
        needed = max(0, source.active_granted + amount - source.retained_quota)
        _expect(
            needed <= state.global_allocatable,
            seq,
            source_id,
            "global_allocatable_after",
            f">= {needed}",
            state.global_allocatable,
            "global allocatable sufficiency",
        )
        expected_expiry = event.mono_ms + state.lease_duration_ms
        _expect_payload_equal(event.payload, "issued_at_ms", event.mono_ms, seq, source_id, "issue timestamp")
        _expect_payload_equal(event.payload, "expires_at_ms", expected_expiry, seq, source_id, "issue expiry")
        _expect_payload_equal(
            event.payload, "quota_from_global", needed, seq, source_id, "global quota draw"
        )
        if event.lease_id is None:
            _mismatch(seq, source_id, "lease_id", "non-null", None, "issue lease id")
        _expect(
            event.lease_id not in source.active,
            seq,
            source_id,
            "lease_id",
            "unique active lease id",
            event.lease_id,
            "duplicate active lease id",
        )
        lease = ReplayLease(
            lease_id=str(event.lease_id),
            source_id=source.source_id,
            amount=amount,
            issued_at_ms=event.mono_ms,
            expires_at_ms=expected_expiry,
        )
        source.retained_quota += needed
        state.global_allocatable -= needed
        source.active[lease.lease_id] = lease
        source.low_streak = 0

    elif event_type == "LEASE_RENEWED":
        source, lease = _find_lease(state, event.lease_id, seq)
        _expect(
            event.amount == lease.amount,
            seq,
            source_id,
            "amount",
            lease.amount,
            event.amount,
            "renewal amount must match the active lease",
        )
        _expect(source.source_id == source_id, seq, source_id, "source_id", source.source_id, source_id, "holder match")
        _expect(
            event.mono_ms < lease.expires_at_ms,
            seq,
            source_id,
            "new_expires_at_ms",
            f"renew before {lease.expires_at_ms}",
            event.mono_ms,
            "renewal must precede expiry",
        )
        old_expires = lease.expires_at_ms
        new_expires = event.mono_ms + state.lease_duration_ms
        _expect_payload_equal(event.payload, "old_expires_at_ms", old_expires, seq, source_id, "old expiry")
        _expect_payload_equal(event.payload, "new_expires_at_ms", new_expires, seq, source_id, "new expiry")
        lease.expires_at_ms = new_expires
        lease.renew_count += 1
        lease.last_renewed_at_ms = event.mono_ms
        _expect_payload_equal(event.payload, "renew_count", lease.renew_count, seq, source_id, "renewal count")

    elif event_type in {"LEASE_EXPIRED", "LEASE_RELEASED"}:
        source, lease = _find_lease(state, event.lease_id, seq)
        _expect(
            event.amount == lease.amount,
            seq,
            source_id,
            "amount",
            lease.amount,
            event.amount,
            "release amount must match the active lease",
        )
        _expect(source.source_id == source_id, seq, source_id, "source_id", source.source_id, source_id, "lease owner")
        _expect_payload_equal(event.payload, "released_amount", lease.amount, seq, source_id, "released amount")
        if event_type == "LEASE_EXPIRED":
            _expect(
                event.mono_ms >= lease.expires_at_ms,
                seq,
                source_id,
                "mono_ms",
                f">= {lease.expires_at_ms}",
                event.mono_ms,
                "expiry requires monotonic deadline",
            )
        source.active.pop(lease.lease_id)
        source.low_streak = 0
        source.last_release_at_ms = event.mono_ms

    elif event_type == "OBSERVED":
        source = _source_by_id(state, source_id, seq)
        active = source.active_granted
        _expect(
            event.amount == active,
            seq,
            source_id,
            "amount",
            active,
            event.amount,
            "observation amount must equal active grants",
        )
        ratio = (active * state.watermark_scale) // source.lease_share if source.lease_share > 0 else 0
        below = ratio < state.reclaim_threshold
        new_streak = source.low_streak + 1 if below else 0
        source.low_streak = new_streak
        source.observation_count += 1
        source.last_ratio_ppm = ratio
        source.last_observed_at_ms = event.mono_ms
        observation = {
            "observation_index": source.observation_count,
            "at_ms": event.mono_ms,
            "active_granted": active,
            "lease_share": source.lease_share,
            "watermark_ratio_ppm": ratio,
            "below_threshold": below,
            "low_streak_after": new_streak,
        }
        source.observation_history.append(observation)
        source.observation_history = source.observation_history[
            -(state.reclaim_window_rounds + 1) :
        ]
        _expect_payload_equal(event.payload, "observation", observation, seq, source_id, "observation window")
        _expect_payload_equal(
            event.payload, "watermark_truncation", WATERMARK_TRUNCATION, seq, source_id, "fixed-point truncation"
        )
        _expect_payload_equal(
            event.payload, "watermark_scale", state.watermark_scale, seq, source_id, "watermark scale"
        )
        _expect_payload_equal(
            event.payload, "reclaim_threshold", state.reclaim_threshold, seq, source_id, "reclaim threshold"
        )
        _expect_payload_equal(
            event.payload,
            "reclaim_window_rounds",
            state.reclaim_window_rounds,
            seq,
            source_id,
            "reclaim rounds",
        )

    elif event_type == "QUOTA_RECLAIMED":
        source = _source_by_id(state, source_id, seq)
        _expect(
            source.low_streak > state.reclaim_window_rounds,
            seq,
            source_id,
            "low_streak",
            f"> {state.reclaim_window_rounds}",
            source.low_streak,
            "consecutive low-water rounds",
        )
        _expect(
            source.last_ratio_ppm < state.reclaim_threshold,
            seq,
            source_id,
            "watermark_ratio_ppm",
            f"< {state.reclaim_threshold}",
            source.last_ratio_ppm,
            "low watermark threshold",
        )
        expected_amount = max(0, source.retained_quota - max(source.reserve_quota, source.active_granted))
        _expect(
            event.amount == expected_amount,
            seq,
            source_id,
            "reclaimed_amount",
            expected_amount,
            event.amount,
            "only idle quota above reserved floor may be reclaimed",
        )
        assert event.amount is not None
        expected_basis = (
            "LOW_WATER:ratio_ppm<threshold;"
            f"consecutive_rounds>{state.reclaim_window_rounds};"
            "reclaim_only_idle_above_reserve"
        )
        _expect_payload_equal(event.payload, "basis", expected_basis, seq, source_id, "quota reclaim basis")
        _expect_payload_equal(
            event.payload, "reclaimed_amount", event.amount, seq, source_id, "quota reclaim amount"
        )
        _expect_payload_equal(
            event.payload,
            "retained_quota_before",
            source.retained_quota,
            seq,
            source_id,
            "retained quota before reclaim",
        )
        _expect_payload_equal(
            event.payload,
            "global_allocatable_before",
            state.global_allocatable,
            seq,
            source_id,
            "global allocatable before reclaim",
        )
        _expect_payload_equal(
            event.payload,
            "observation_window",
            source.observation_history,
            seq,
            source_id,
            "quota reclaim observation window",
        )
        source.retained_quota -= event.amount
        state.global_allocatable += event.amount
        source.reclaimed_total += event.amount
        source.last_reclaim_basis = expected_basis

    elif event_type == "RACE_REJECTED":
        _expect_payload_equal(event.payload, "code", "E_RACE", seq, source_id, "race rejection code")
        _expect_payload_equal(event.payload, "ownership_unchanged", True, seq, source_id, "race state immutability")
        basis = event.payload.get("basis", "")
        _expect(
            isinstance(basis, str) and basis.startswith("RACE:"),
            seq,
            source_id,
            "basis",
            "RACE:*",
            basis,
            "race basis",
        )

    elif event_type == "LEASE_UNKNOWN":
        _expect_payload_equal(event.payload, "unknown", True, seq, source_id, "unknown lease marker")
        _expect_payload_equal(
            event.payload, "exit_report_available", False, seq, source_id, "no crash exit report"
        )

    _verify_common_payload(state, event, decision)
    if event_type == "QUOTA_RECLAIMED":
        source = _source_by_id(state, source_id, seq)
        source.last_reclaim_seq = seq


def _verify_common_payload(state: ReplayState, event: EventRecord, decision: str) -> None:
    seq = event.seq
    source_id = event.source_id
    _expect_payload_equal(
        event.payload, "global_allocatable_after", state.global_allocatable, seq, source_id, "global invariant"
    )
    _expect_payload_equal(
        event.payload, "total_active_after", state.total_active, seq, source_id, "capacity invariant"
    )
    _expect_payload_equal(
        event.payload, "retained_total_after", state.retained_total, seq, source_id, "retained quota invariant"
    )
    ownership = state.ownership()
    _expect_payload_equal(event.payload, "ownership", ownership, seq, source_id, "ownership set replay")
    _expect_payload_equal(
        event.payload,
        "ownership_digest",
        canonical_digest(ownership),
        seq,
        source_id,
        "ownership digest replay",
    )
    if event.source_id is not None:
        source = _source_by_id(state, event.source_id, seq)
        _expect_payload_equal(
            event.payload,
            "source_state_after",
            source.as_dict(),
            seq,
            source_id,
            "source state replay",
        )


def replay_ledger(
    ledger_path: str | Path,
    scenario: Scenario | None = None,
) -> dict:
    ledger = AuditLedger(ledger_path, readonly=True, verify=True)
    try:
        meta = ledger.meta()
        integrity = ledger.verify_integrity()
        drift: dict[str, Any] | None = None
        if scenario is not None and scenario.scenario_digest != meta["scenario_digest"]:
            differences = scenario_diff(meta["scenario"], scenario)
            drift = {
                "classification": "EXTERNAL_DRIFT",
                "ledger_scenario_version": meta["scenario_version"],
                "current_scenario_version": scenario.scenario_version,
                "ledger_scenario_digest": meta["scenario_digest"],
                "current_scenario_digest": scenario.scenario_digest,
                "field_differences": differences,
                "ledger_defect": False,
            }

        params = meta["params"]
        state = ReplayState(params)
        for event in ledger.events():
            if event.params_version != meta["params_version"]:
                _mismatch(
                    event.seq,
                    event.source_id,
                    "params_version",
                    meta["params_version"],
                    event.params_version,
                    "ledger parameter version",
                )
            if event.params_digest != meta["params_digest"]:
                _mismatch(
                    event.seq,
                    event.source_id,
                    "params_digest",
                    meta["params_digest"],
                    event.params_digest,
                    "ledger parameter digest",
                )
            if state.last_mono_ms is not None and event.mono_ms < state.last_mono_ms:
                _mismatch(
                    event.seq,
                    event.source_id,
                    "mono_ms",
                    f">= {state.last_mono_ms}",
                    event.mono_ms,
                    "monotonic event ordering",
                )
            state.last_mono_ms = event.mono_ms
            try:
                _apply_event(state, event)
            except ReplayMismatch:
                raise
            except (KeyError, TypeError, ValueError, AssertionError) as exc:
                _mismatch(
                    event.seq,
                    event.source_id,
                    "payload",
                    "well-formed replayable event",
                    f"{type(exc).__name__}: {exc}",
                    "event payload shape",
                )
            if state.retained_total + state.global_allocatable != state.capacity:
                _mismatch(
                    event.seq,
                    event.source_id,
                    "global_allocatable_after",
                    state.capacity - state.retained_total,
                    state.global_allocatable,
                    "retained quota plus global equals capacity",
                )

        result = {
            "status": "EXTERNAL_DRIFT" if drift else "OK",
            "business_code": "E_EXTERNAL_DRIFT" if drift else "OK",
            "ledger_defect": False,
            "format_version": integrity["format_version"],
            "content_digest": integrity["content_digest"],
            "event_count": integrity["event_count"],
            "params_version": integrity["params_version"],
            "params_digest": integrity["params_digest"],
            "final_state": state.final_state(),
            "external_drift": drift,
        }
        return result
    finally:
        ledger.close()

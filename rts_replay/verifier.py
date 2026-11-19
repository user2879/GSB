"""Cross-path reconciliation and RNG drift detection.

The verifier compares the online frame-driven path against the authoritative
offline path tick by tick. Consistency is established on full state hashes
(never on resource totals alone) and on the ordered ledger entries. The first
diverging tick is reported together with its event context and participating
actors.

RNG drift is demonstrated by re-synthesizing with an altered derivation chain:
every tick that consumed randomness must change hash, and the first such tick is
flagged with ``E_RNG_DRIFT``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine import TickEngine
from .errors import ErrorCode, RtsError, TickError
from .events import Event
from .hashing import canonical_hash
from .offline import OfflineResult
from .player import ReplayPlayer
from .rules import RuleConfig


@dataclass
class Mismatch:
    tick: int
    expected_hash: str
    actual_hash: str
    actors: list[str]
    events: list[dict]

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "actors": self.actors,
            "events": self.events,
        }


@dataclass
class ReconcileResult:
    ok: bool
    tick_count: int
    ledger_match: bool
    first_mismatch: Mismatch | None = None
    rng_drift_tick: int | None = None
    rng_drift_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tick_count": self.tick_count,
            "ledger_match": self.ledger_match,
            "first_mismatch": (
                self.first_mismatch.to_dict() if self.first_mismatch else None
            ),
            "rng_drift_tick": self.rng_drift_tick,
            "rng_drift_events": self.rng_drift_events,
        }


def _events_at(events: list[Event], tick: int) -> list[Event]:
    return [event for event in events if event.tick == tick]


def reconcile(
    config: RuleConfig,
    events: list[Event],
    offline: OfflineResult,
    *,
    budget_ms: float = 16.0,
    base_ticks_per_frame: int = 4,
) -> ReconcileResult:
    online = ReplayPlayer(
        config,
        events,
        budget_ms=budget_ms,
        base_ticks_per_frame=base_ticks_per_frame,
    )
    online.run_to_end()

    online_hashes = {record.tick: record.state_hash for record in online.engine.records}
    offline_hashes = {record['tick']: record['state_hash'] for record in offline.tick_hashes}
    if set(online_hashes) != set(offline_hashes):
        raise RtsError(
            ErrorCode.E_INPUT_MALFORMED,
            "online and offline produced different tick sets",
            context={
                "online_ticks": sorted(online_hashes),
                "offline_ticks": sorted(offline_hashes),
            },
        )

    first_mismatch: Mismatch | None = None
    for tick in sorted(offline_hashes):
        if online_hashes[tick] != offline_hashes[tick]:
            tick_events = _events_at(events, tick)
            first_mismatch = Mismatch(
                tick=tick,
                expected_hash=offline_hashes[tick],
                actual_hash=online_hashes[tick],
                actors=sorted({event.actor for event in tick_events}),
                events=[event.to_dict() for event in tick_events],
            )
            break

    online_ledger = [entry.to_dict() for entry in online.engine.ledger]
    ledger_match = canonical_hash(online_ledger) == canonical_hash(offline.ledger)

    drift_tick, drift_events = _detect_rng_drift(config, events, offline)

    ok = first_mismatch is None and ledger_match and drift_tick is not None
    return ReconcileResult(
        ok=ok,
        tick_count=len(offline_hashes),
        ledger_match=ledger_match,
        first_mismatch=first_mismatch,
        rng_drift_tick=drift_tick,
        rng_drift_events=drift_events,
    )


def _detect_rng_drift(
    config: RuleConfig, events: list[Event], offline: OfflineResult
) -> tuple[int | None, list[dict]]:
    drifted = TickEngine(
        config, events, chain_version=config.rng_chain_version + 1
    )
    drifted.advance_to(drifted.last_tick)
    baseline = {record['tick']: record['state_hash'] for record in offline.tick_hashes}
    for record in drifted.records:
        if baseline.get(record.tick) != record.state_hash:

            tick_events = _events_at(events, record.tick)
            return record.tick, [event.to_dict() for event in tick_events]
    return None, []


def verify_jump_equivalence(
    config: RuleConfig,
    events: list[Event],
    offline: OfflineResult,
    jump_ticks: list[int],
) -> dict:
    """Every jump must rebuild state equal to continuous synthesis at that tick."""
    baseline = {0: offline.engine.snapshots[0].state_hash}
    baseline.update(
        {record['tick']: record['state_hash'] for record in offline.tick_hashes}
    )
    checks = []
    for target in jump_ticks:
        player = ReplayPlayer(config, events)
        player.jump_to(target)
        actual = player.engine.state.state_hash()
        checks.append(
            {
                "target_tick": target,
                "expected_hash": baseline.get(target),
                "actual_hash": actual,
                "match": actual == baseline.get(target),
            }
        )
    return {"all_match": all(item["match"] for item in checks), "checks": checks}


def error_code_audit(offline: OfflineResult) -> dict:
    counts: dict[str, int] = {}
    first_locations: dict[str, dict] = {}
    for error in offline.errors:
        counts[error["code"]] = counts.get(error["code"], 0) + 1
        first_locations.setdefault(
            error["code"],
            {"tick": error["tick"], "actor": error["actor"], "message": error["message"]},
        )
    return {
        "error_counts": dict(sorted(counts.items())),
        "first_locations": dict(sorted(first_locations.items())),
    }

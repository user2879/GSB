"""Deterministic readiness/freeze arbitration (requirement 2).

A shard stays at the frozen readiness edge until *every* upstream shard has a
committed outcome *and* an available aggregate summary.  All iteration is over
sorted identifier lists, so a round's verdict never depends on dict/hash
order.  Frozen shards are never dispatched and never hold a lease.
"""

from threading import Lock
from typing import Dict, Iterable, List, Optional, Set

from .clock import Clock

FROZEN = "FROZEN"
UNFROZEN = "UNFROZEN"
UPSTREAM_HELD = "UPSTREAM_HELD"  # terminal upstreams but summary unavailable


class FreezeManager:
    def __init__(self, clock: Clock, freeze_threshold_ms: int):
        self._clock = clock
        self._threshold = freeze_threshold_ms
        self._lock = Lock()
        self._state: Dict[str, str] = {}
        self._since: Dict[str, int] = {}
        self._missing: Dict[str, List[str]] = {}
        self._history: List[dict] = []

    def initialize(self, shard_ids: Iterable[str]):
        now = self._clock.now_ms()
        with self._lock:
            for shard_id in sorted(shard_ids):
                self._state[shard_id] = FROZEN
                self._since[shard_id] = now
                self._missing[shard_id] = []

    def state_of(self, shard_id: str) -> Optional[str]:
        return self._state.get(shard_id)

    def is_frozen(self, shard_id: str) -> bool:
        return self._state.get(shard_id) in (FROZEN, UPSTREAM_HELD)

    def frozen_shards(self) -> List[str]:
        return sorted(sid for sid, state in self._state.items() if state in (FROZEN, UPSTREAM_HELD))

    def ready_shards(self) -> List[str]:
        return sorted(sid for sid, state in self._state.items() if state == UNFROZEN)

    def missing_upstreams(self, shard_id: str) -> List[str]:
        return list(self._missing.get(shard_id, []))

    def history(self) -> List[dict]:
        return list(self._history)

    def evaluate_round(self, shards, committed_outcomes: Dict[str, dict]) -> List[dict]:
        """Re-evaluate every shard once; return the migration records of the round.

        ``committed_outcomes`` maps shard_id to a committed outcome dict that
        must contain a non-null ``summary_digest`` to count as available.
        """
        now = self._clock.now_ms()
        migrations: List[dict] = []
        # Fixed order: topological callers pass sorted/topo ids; we sort again
        # here so the contract holds regardless of the input container order.
        ordered = sorted(shards, key=lambda s: s.shard_id)
        with self._lock:
            for shard in ordered:
                if self._state.get(shard.shard_id) == UNFROZEN:
                    continue
                missing = self._missing_upstreams_locked(shard, committed_outcomes)
                if not missing:
                    prior = self._state[shard.shard_id]
                    self._state[shard.shard_id] = UNFROZEN
                    self._missing[shard.shard_id] = []
                    record = self._record(shard.shard_id, prior, UNFROZEN, [], now)
                    migrations.append(record)
                else:
                    prior = self._state.get(shard.shard_id, FROZEN)
                    self._state[shard.shard_id] = UPSTREAM_HELD
                    self._missing[shard.shard_id] = missing
                    record = self._record(shard.shard_id, prior, UPSTREAM_HELD, missing, now)
                    migrations.append(record)
        return migrations

    def _missing_upstreams_locked(self, shard, committed_outcomes) -> List[str]:
        missing: List[str] = []
        for dep in sorted(shard.depends_on):
            outcome = committed_outcomes.get(dep)
            if outcome is None or not outcome.get("summary_digest"):
                missing.append(dep)
        return missing

    def refreeze(self, shard_id: str, reason: str, missing: Optional[List[str]] = None) -> dict:
        """Push a shard back behind the readiness edge (e.g. invalidation)."""
        now = self._clock.now_ms()
        with self._lock:
            prior = self._state.get(shard_id, FROZEN)
            self._state[shard_id] = UPSTREAM_HELD if reason == "upstream" else FROZEN
            self._since[shard_id] = now
            self._missing[shard_id] = list(missing or [])
            return self._record(shard_id, prior, self._state[shard_id], self._missing[shard_id], now, reason)

    def _record(self, shard_id, prior, new_state, missing, now, reason="round_evaluation") -> dict:
        waited = now - self._since.get(shard_id, now)
        breached = self._threshold >= 0 and waited > self._threshold
        record = {
            "shard_id": shard_id,
            "from_state": prior,
            "to_state": new_state,
            "missing_upstreams": list(missing),
            "at_ms": now,
            "frozen_elapsed_ms": waited,
            "threshold_ms": self._threshold,
            "threshold_breached": bool(breached),
            "reason": reason,
        }
        self._history.append(record)
        if new_state == UNFROZEN:
            self._since[shard_id] = now
        return record

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": dict(self._state),
                "since": dict(self._since),
                "missing": {k: list(v) for k, v in self._missing.items()},
                "history": list(self._history),
            }

    def restore_snapshot(self, snapshot: dict):
        with self._lock:
            self._state = dict(snapshot.get("state", {}))
            self._since = dict(snapshot.get("since", {}))
            self._missing = {k: list(v) for k, v in snapshot.get("missing", {}).items()}
            self._history = list(snapshot.get("history", []))

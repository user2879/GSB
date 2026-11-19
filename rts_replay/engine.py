"""Tick-by-tick deterministic state synthesis engine.

The engine processes events grouped by tick and ordered inside each tick by
``(actor, seq)``. It never looks past the current tick, so every emitted
snapshot depends only on information available up to that tick (no look-ahead).
Hard ordering/input problems raise :class:`RtsError`; soft business violations
(resource overflow, blocked queues, blocked moves) are collected as
:class:`TickError` records so synthesis and reporting continue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ErrorCode, RtsError, TickError
from .events import (
    ACTION_COLLECT,
    ACTION_MOVE,
    ACTION_PRODUCE,
    Event,
    sort_key,
)
from .rng import RNG_CHAIN_VERSION, draw_bounded, make_event_rng
from .rules import RuleConfig
from .state import GameState, QueueItem, Snapshot, build_snapshot


@dataclass(frozen=True)
class LedgerEntry:
    tick: int
    actor: str
    seq: int
    action: str
    target: str
    amount: int
    rng_seed_delta: int
    rng_value: int
    resources_before: int
    resources_after: int
    result: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "actor": self.actor,
            "seq": self.seq,
            "action": self.action,
            "target": self.target,
            "amount": self.amount,
            "rng_seed_delta": self.rng_seed_delta,
            "rng_value": self.rng_value,
            "resources_before": self.resources_before,
            "resources_after": self.resources_after,
            "result": self.result,
            "note": self.note,
        }


@dataclass
class TickRecord:
    tick: int
    state_hash: str
    event_ids: list[tuple[str, int]]
    errors: list[TickError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "state_hash": self.state_hash,
            "event_ids": [list(item) for item in self.event_ids],
            "errors": [item.to_dict() for item in self.errors],
        }


class TickEngine:
    """Incremental synthesis engine usable by both offline and online paths."""

    def __init__(
        self,
        config: RuleConfig,
        events: list[Event],
        *,
        chain_version: int | None = None,
        snapshot_interval: int = 10,
    ) -> None:
        config.validate()
        self.config = config
        self.events = list(events)
        self.chain_version = (
            config.rng_chain_version if chain_version is None else chain_version
        )
        self.snapshot_interval = snapshot_interval
        self.state = GameState.initial(config)
        self.ledger: list[LedgerEntry] = []
        self.records: list[TickRecord] = []
        self.snapshots: dict[int, Snapshot] = {0: build_snapshot(self.state)}
        self._groups = self._group_events(events)
        self._tick_order = sorted(self._groups)
        self._pending: list[Event] = []
        self._cursor = 0
        self.last_tick = self._tick_order[-1] if self._tick_order else 0

    def _group_events(self, events: list[Event]) -> dict[int, list[Event]]:
        ordered = sorted(events, key=sort_key)
        groups: dict[int, list[Event]] = {}
        for event in ordered:
            if event.actor not in self.config.faction_names():
                raise RtsError(
                    ErrorCode.E_INPUT_MALFORMED,
                    f"event actor '{event.actor}' has no rule configuration",
                    tick=event.tick,
                    actor=event.actor,
                )
            groups.setdefault(event.tick, []).append(event)
        return groups

    def advance_to(self, target_tick: int) -> GameState:
        """Synthesize every remaining tick up to and including *target_tick*."""
        while self._cursor < len(self._groups) and self._pending_tick() <= target_tick:
            self._step_next_tick()
        return self.state

    def step(self) -> GameState:
        if self._cursor >= len(self._groups):
            return self.state
        self._step_next_tick()
        return self.state

    def _pending_tick(self) -> int:
        return self._tick_order[self._cursor]

    def _step_next_tick(self) -> None:
        tick = self._tick_order[self._cursor]
        tick_events = self._groups[tick]
        self._cursor += 1
        self._pending = tick_events
        record = TickRecord(
            tick=tick, state_hash="", event_ids=[(e.actor, e.seq) for e in tick_events]
        )
        self.state.tick = tick

        for event in tick_events:
            self._check_sequence(event, record)
            self.state.last_seq[event.actor] = event.seq
            self._dispatch(event, record)

        for faction in self.config.faction_names():
            self._advance_queue(tick, faction)

        self.state.tick = tick
        record.state_hash = self.state.state_hash()
        self.records.append(record)

        if tick % self.snapshot_interval == 0 or tick == self.last_tick:
            self.snapshots[tick] = build_snapshot(self.state)

    def _check_sequence(self, event: Event, record: TickRecord) -> None:
        expected = self.state.last_seq[event.actor] + 1
        if event.seq != expected:
            record.errors.append(
                TickError(
                    code=ErrorCode.E_SEQ_GAP.value,
                    message=(
                        f"actor '{event.actor}' sequence gap at tick {event.tick}: "
                        f"expected seq {expected}, got {event.seq}"
                    ),
                    tick=event.tick,
                    actor=event.actor,
                    action=event.action,
                    seq=event.seq,
                    context={"expected": expected, "actual": event.seq},
                )
            )

    def _dispatch(self, event: Event, record: TickRecord) -> None:
        rng = make_event_rng(
            self.config.match_seed,
            event.tick,
            event.rng_seed_delta,
            self.chain_version,
        )
        if event.action == ACTION_COLLECT:
            self._handle_collect(event, rng, record)
        elif event.action == ACTION_PRODUCE:
            self._handle_produce(event, rng, record)
        elif event.action == ACTION_MOVE:
            self._handle_move(event, rng, record)
        else:  # validated earlier, kept defensive
            raise RtsError(
                ErrorCode.E_INPUT_MALFORMED,
                f"unsupported action '{event.action}'",
                tick=event.tick,
                actor=event.actor,
            )

    def _handle_collect(self, event: Event, rng, record: TickRecord) -> None:
        rules = self.config.get(event.actor)
        faction = self.state.factions[event.actor]
        before = faction.resources
        rng_value = draw_bounded(rng, -5, 5)
        gained = max(0, event.amount + rng_value)
        projected = before + gained
        result = "ok"
        note = ""
        if projected > rules.resource_cap:
            gained = max(0, rules.resource_cap - before)
            result = "clamped"
            note = "collect exceeded resource cap"
            record.errors.append(
                TickError(
                    code=ErrorCode.E_RESOURCE_OVERFLOW.value,
                    message=(
                        f"actor '{event.actor}' resource overflow at tick {event.tick}: "
                        f"projected {projected} > cap {rules.resource_cap}"
                    ),
                    tick=event.tick,
                    actor=event.actor,
                    action=event.action,
                    seq=event.seq,
                    context={
                        "projected": projected,
                        "cap": rules.resource_cap,
                        "clamped_to": before + gained,
                    },
                )
            )
        faction.resources = before + gained
        faction.total_collected += gained
        self.ledger.append(
            LedgerEntry(
                tick=event.tick,
                actor=event.actor,
                seq=event.seq,
                action=event.action,
                target=event.target,
                amount=event.amount,
                rng_seed_delta=event.rng_seed_delta,
                rng_value=rng_value,
                resources_before=before,
                resources_after=faction.resources,
                result=result,
                note=note,
            )
        )

    def _handle_produce(self, event: Event, rng, record: TickRecord) -> None:
        rules = self.config.get(event.actor)
        faction = self.state.factions[event.actor]
        before = faction.resources
        rng_value = draw_bounded(rng, 1, 3)
        cost = rules.unit_cost * event.amount

        if faction.active_commands >= rules.max_concurrent_commands:
            record.errors.append(
                TickError(
                    code=ErrorCode.E_QUEUE_BLOCKED.value,
                    message=(
                        f"actor '{event.actor}' production blocked at tick {event.tick}: "
                        f"{faction.active_commands} active commands >= limit "
                        f"{rules.max_concurrent_commands}"
                    ),
                    tick=event.tick,
                    actor=event.actor,
                    action=event.action,
                    seq=event.seq,
                    context={
                        "active_commands": faction.active_commands,
                        "limit": rules.max_concurrent_commands,
                    },
                )
            )
            self._append_blocked_entry(event, rng_value, before, "blocked_concurrency")
            return

        if before < cost:
            record.errors.append(
                TickError(
                    code=ErrorCode.E_QUEUE_BLOCKED.value,
                    message=(
                        f"actor '{event.actor}' production blocked at tick {event.tick}: "
                        f"resources {before} < cost {cost}"
                    ),
                    tick=event.tick,
                    actor=event.actor,
                    action=event.action,
                    seq=event.seq,
                    context={"resources": before, "cost": cost},
                )
            )
            self._append_blocked_entry(event, rng_value, before, "blocked_resources")
            return

        faction.resources = before - cost
        faction.total_spent += cost
        faction.active_commands += 1
        item = QueueItem(
            order_id=faction.next_order_id,
            target=event.target,
            amount=event.amount,
            enqueue_tick=event.tick,
            ready_tick=event.tick + self.config.queue_ticks,
            quality=rng_value,
        )
        faction.next_order_id += 1
        faction.queue.append(item)
        self.ledger.append(
            LedgerEntry(
                tick=event.tick,
                actor=event.actor,
                seq=event.seq,
                action=event.action,
                target=event.target,
                amount=event.amount,
                rng_seed_delta=event.rng_seed_delta,
                rng_value=rng_value,
                resources_before=before,
                resources_after=faction.resources,
                result="enqueued",
                note=f"order_id={item.order_id},ready_tick={item.ready_tick}",
            )
        )

    def _handle_move(self, event: Event, rng, record: TickRecord) -> None:
        faction = self.state.factions[event.actor]
        before = faction.resources
        rng_value = draw_bounded(rng, 0, 100)
        if faction.idle_units <= 0:
            record.errors.append(
                TickError(
                    code=ErrorCode.E_QUEUE_BLOCKED.value,
                    message=(
                        f"actor '{event.actor}' move blocked at tick {event.tick}: "
                        "no idle units available"
                    ),
                    tick=event.tick,
                    actor=event.actor,
                    action=event.action,
                    seq=event.seq,
                    context={"idle_units": faction.idle_units, "requested": event.amount},
                )
            )
            self._append_blocked_entry(event, rng_value, before, "blocked_no_units")
            return

        dispatch = min(event.amount, faction.idle_units)
        losses = dispatch if rng_value == 0 else 0
        faction.idle_units -= dispatch
        faction.deployed_units += dispatch - losses
        note = "ok" if losses == 0 else f"lost {losses} units (rng_value=0)"
        self.ledger.append(
            LedgerEntry(
                tick=event.tick,
                actor=event.actor,
                seq=event.seq,
                action=event.action,
                target=event.target,
                amount=event.amount,
                rng_seed_delta=event.rng_seed_delta,
                rng_value=rng_value,
                resources_before=before,
                resources_after=before,
                result="moved" if losses == 0 else "moved_with_losses",
                note=note,
            )
        )

    def _append_blocked_entry(
        self, event: Event, rng_value: int, resources: int, result: str
    ) -> None:
        self.ledger.append(
            LedgerEntry(
                tick=event.tick,
                actor=event.actor,
                seq=event.seq,
                action=event.action,
                target=event.target,
                amount=event.amount,
                rng_seed_delta=event.rng_seed_delta,
                rng_value=rng_value,
                resources_before=resources,
                resources_after=resources,
                result=result,
            )
        )

    def _advance_queue(self, tick: int, faction_name: str) -> None:
        faction = self.state.factions[faction_name]
        remaining: list[QueueItem] = []
        for item in faction.queue:
            if item.ready_tick <= tick:
                faction.idle_units += item.amount
                faction.active_commands = max(0, faction.active_commands - 1)
            else:
                remaining.append(item)
        faction.queue = remaining

    def all_errors(self) -> list[TickError]:
        return [error for record in self.records for error in record.errors]

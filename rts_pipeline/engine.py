"""Causal tick-range state synthesis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import hash_value
from .constants import ENGINE_VERSION
from .errors import PipelineError, require
from .events import normalize_events
from .rng import derive_record
from .rules import clone_state, faction_hash, validate_state


def _require_tick_range(start_tick: int, end_tick: int) -> None:
    require(
        isinstance(start_tick, int)
        and not isinstance(start_tick, bool)
        and isinstance(end_tick, int)
        and not isinstance(end_tick, bool)
        and start_tick >= 0
        and end_tick >= start_tick,
        "E_SCHEMA",
        "invalid tick range",
        start_tick=start_tick,
        end_tick=end_tick,
    )


def _advance_queues(state: dict[str, Any]) -> None:
    for faction in sorted(state["factions"]):
        for item in state["factions"][faction]["queue"]:
            if item["remaining"] > 0:
                item["remaining"] -= 1


def _state_hash(snapshot: dict[str, Any]) -> str:
    return hash_value(snapshot)


def _violation(event: dict[str, Any], reason: str, **details: Any) -> dict[str, Any]:
    return {
        "code": "E_RESOURCE_OVERFLOW",
        "reason": reason,
        "tick": event["tick"],
        "actor": event["actor"],
        "local_seq": event["local_seq"],
        "action": event["action"],
        "target": event["target"],
        "amount": event["amount"],
        **details,
    }


def _apply_event(
    state: dict[str, Any],
    event: dict[str, Any],
    rules: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    faction = event["actor"]
    actor_state = state["factions"][faction]
    body = rules["factions"][faction]
    rng = derive_record(seed, event["tick"], event["rng_seed_delta"])
    violation: dict[str, Any] | None = None
    applied = False
    resource_delta = 0
    queue_delta = 0

    if event["action"] == "collect":
        gain = event["amount"] * body["collect_rate"] + (int(rng["roll_10000"]) % body["collect_rate"])
        attempted = actor_state["resources"] + gain
        if attempted > body["resource_cap"]:
            violation = _violation(
                event,
                "resource_cap",
                resource_cap=body["resource_cap"],
                resources_before=actor_state["resources"],
                attempted_resources=attempted,
                attempted_gain=gain,
            )
        else:
            actor_state["resources"] = attempted
            actor_state["collected_total"] += gain
            resource_delta = gain
            applied = True
    elif event["action"] == "produce":
        unit = event["target"]
        cost = body["unit_costs"][unit] * event["amount"]
        queue_after = len(actor_state["queue"]) + event["amount"]
        if actor_state["resources"] - cost < 0:
            violation = _violation(
                event,
                "resource_negative",
                resources_before=actor_state["resources"],
                attempted_resources=actor_state["resources"] - cost,
                attempted_cost=cost,
            )
        elif queue_after > body["queue_capacity"]:
            violation = _violation(
                event,
                "queue_capacity",
                queue_capacity=body["queue_capacity"],
                queue_depth_before=len(actor_state["queue"]),
                attempted_queue_depth=queue_after,
            )
        else:
            actor_state["resources"] -= cost
            actor_state["spent_total"] += cost
            actor_state["produced_total"] += event["amount"]
            resource_delta = -cost
            queue_delta = event["amount"]
            remaining = body["build_ticks"][unit]
            actor_state["queue"].extend(
                {"unit": unit, "remaining": remaining} for _ in range(event["amount"])
            )
            applied = True
    elif event["action"] == "dispatch":
        unit = event["target"]
        ready_indices = [
            index
            for index, item in enumerate(actor_state["queue"])
            if item["unit"] == unit and item["remaining"] == 0
        ]
        if len(ready_indices) < event["amount"]:
            violation = _violation(
                event,
                "queue_underflow",
                unit=unit,
                requested_units=event["amount"],
                ready_units=len(ready_indices),
            )
        else:
            remove = set(ready_indices[: event["amount"]])
            actor_state["queue"] = [
                item for index, item in enumerate(actor_state["queue"]) if index not in remove
            ]
            actor_state["dispatched_total"] += event["amount"]
            queue_delta = -event["amount"]
            applied = True
    else:
        raise PipelineError("E_MALFORMED_EVENT", "unknown action during synthesis", action=event["action"])

    state["event_cursors"][faction] += 1
    event_record = {
        "tick": event["tick"],
        "actor": faction,
        "local_seq": event["local_seq"],
        "action": event["action"],
        "target": event["target"],
        "amount": event["amount"],
        "rng_seed_delta": event["rng_seed_delta"],
        "rng": rng,
        "outcome": "applied" if applied else "rejected",
        "violation_reason": None if violation is None else violation["reason"],
        "resource_delta": resource_delta,
        "queue_delta": queue_delta,
        "state_after": deepcopy(actor_state),
        "resource_queue_hash": faction_hash(actor_state),
    }
    return event_record, violation


def _validate_and_plan(
    events: list[dict[str, Any]],
    rules: dict[str, Any],
    state: dict[str, Any],
    start_tick: int,
    end_tick: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    by_tick: dict[int, list[dict[str, Any]]] = {}
    cursors = dict(state["event_cursors"])
    for event in events:
        if event["tick"] < start_tick or event["tick"] > end_tick:
            raise PipelineError(
                "E_MALFORMED_EVENT",
                "event lies outside the requested tick range",
                tick=event["tick"],
                start_tick=start_tick,
                end_tick=end_tick,
            )
        expected = cursors[event["actor"]]
        if event["local_seq"] != expected:
            raise PipelineError(
                "E_SEQ_GAP",
                "local sequence is not contiguous for actor",
                actor=event["actor"],
                tick=event["tick"],
                expected_local_seq=expected,
                actual_local_seq=event["local_seq"],
            )
        cursors[event["actor"]] += 1
        by_tick.setdefault(event["tick"], []).append(event)
    for tick_events in by_tick.values():
        tick_events.sort(key=lambda item: (item["actor"], item["local_seq"]))
    return by_tick, cursors


def synthesize_range(
    events: list[dict[str, Any]],
    rules: dict[str, Any],
    seed: int,
    start_tick: int,
    end_tick: int,
    entry_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Synthesize a closed tick range from an explicit entry snapshot.

    The entry snapshot is the committed state after ``start_tick - 1`` and
    includes per-actor event cursors. This makes chunks independent without
    weakening local sequence validation.
    """

    _require_tick_range(start_tick, end_tick)
    require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "E_SCHEMA",
        "game seed must be a non-negative integer",
        seed=seed,
    )
    state = validate_state(entry_snapshot, rules)
    normalized = normalize_events(events, rules)
    by_tick, planned_cursors = _validate_and_plan(normalized, rules, state, start_tick, end_tick)
    tick_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []

    for tick in range(start_tick, end_tick + 1):
        _advance_queues(state)
        tick_events = by_tick.get(tick, [])
        for event in tick_events:
            event_record, violation = _apply_event(state, event, rules, seed)
            event_records.append(event_record)
            if violation is not None:
                violations.append(violation)
        faction_hashes = {
            faction: faction_hash(state["factions"][faction])
            for faction in sorted(state["factions"])
        }
        snapshot = clone_state(state)
        tick_records.append(
            {
                "tick": tick,
                "snapshot": snapshot,
                "faction_hashes": faction_hashes,
                "state_hash": _state_hash(snapshot),
                "event_keys": [
                    {
                        "actor": event["actor"],
                        "local_seq": event["local_seq"],
                    }
                    for event in tick_events
                ],
            }
        )

    if state["event_cursors"] != planned_cursors:
        raise PipelineError("E_SEQ_GAP", "event cursor did not advance as planned")
    exit_snapshot = clone_state(state)
    return {
        "engine_version": ENGINE_VERSION,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "entry_snapshot": clone_state(entry_snapshot),
        "entry_snapshot_hash": hash_value(entry_snapshot),
        "tick_records": tick_records,
        "event_records": event_records,
        "violations": violations,
        "rng_records": [record["rng"] for record in event_records],
        "exit_snapshot": exit_snapshot,
        "exit_snapshot_hash": hash_value(exit_snapshot),
        "result_codes": ["E_RESOURCE_OVERFLOW"] if violations else [],
        "event_count": len(event_records),
    }


def synthesize_game(
    events: list[dict[str, Any]],
    rules: dict[str, Any],
    seed: int,
    tick_start: int,
    tick_end: int,
    initial_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    return synthesize_range(
        events=events,
        rules=rules,
        seed=seed,
        start_tick=tick_start,
        end_tick=tick_end,
        entry_snapshot=initial_snapshot,
    )

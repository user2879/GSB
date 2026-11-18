from __future__ import annotations

import copy
from typing import Any

from .loading import make_failure
from .merge import merge_slot_update, parse_slot_value
from .util import (
    PipelineValidationError,
    business_code,
    stable_hash,
    transition_event_hash,
)


def state_entry_count(slots: dict[str, Any]) -> int:
    """One session record plus one entry per materialized slot."""
    return 1 + len(slots)


def _clone_slots(slots: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(slots)


def _condition_matches(
    slot_name: str,
    condition: dict[str, Any],
    slots: dict[str, Any],
    slot_schema: dict[str, Any],
) -> bool:
    operation = condition.get("op")
    if operation == "exists":
        return slot_name in slots
    if operation == "not_exists":
        return slot_name not in slots
    if slot_name not in slots:
        return False
    actual = slots[slot_name]["value"]
    _, comparable_actual = parse_slot_value(slot_schema, actual)
    if operation == "in":
        candidates = condition.get("value")
        if not isinstance(candidates, list):
            raise PipelineValidationError("'in' condition requires a list value")
        for candidate in candidates:
            _, comparable_candidate = parse_slot_value(slot_schema, candidate)
            if comparable_actual == comparable_candidate:
                return True
        return False
    _, comparable_expected = parse_slot_value(slot_schema, condition.get("value"))
    if operation == "eq":
        return comparable_actual == comparable_expected
    if operation == "ne":
        return comparable_actual != comparable_expected
    if operation == "gt":
        return comparable_actual > comparable_expected
    if operation == "gte":
        return comparable_actual >= comparable_expected
    if operation == "lt":
        return comparable_actual < comparable_expected
    if operation == "lte":
        return comparable_actual <= comparable_expected
    raise PipelineValidationError(f"unsupported condition operation {operation!r}")


def rule_matches(
    rule: dict[str, Any],
    current_state: str,
    slots: dict[str, Any],
    event: dict[str, Any],
    machine: dict[str, Any],
) -> bool:
    source = rule.get("from")
    if source != "*" and source != current_state:
        return False
    roles = rule.get("roles")
    if roles and event.get("role") not in roles:
        return False
    text_contains = rule.get("text_contains")
    if text_contains is not None and text_contains not in str(event.get("text", "")):
        return False
    for slot_name, condition in rule.get("when", {}).items():
        if not _condition_matches(slot_name, condition, slots, machine["slots"][slot_name]):
            return False
    return True


def _candidate_rules(
    machine: dict[str, Any],
    current_state: str,
    slots: dict[str, Any],
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    matches = [
        rule
        for rule in machine.get("transitions", [])
        if rule_matches(rule, current_state, slots, event, machine)
    ]
    return sorted(matches, key=lambda rule: rule["id"])


def _transition_hash(
    *,
    turn: int,
    source_state: str,
    target_state: str,
    rule_id: str,
    candidates: list[str],
    event: dict[str, Any],
    machine: dict[str, Any],
    state_cache_key: str,
) -> str:
    relevant_rules = [
        rule for rule in machine.get("transitions", []) if rule.get("from") in {source_state, "*"}
    ]
    return stable_hash(
        {
            "kind": "transition",
            "turn": turn,
            "from": source_state,
            "to": target_state,
            "rule_id": rule_id,
            "candidates": candidates,
            "event": transition_event_hash(event),
            "transition_schema": relevant_rules,
            "state_cache_key": state_cache_key,
        }
    )


def _state_cache_key(
    *,
    session_id: str,
    current_state: str,
    slots: dict[str, Any],
    transition_count: int,
    machine: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "kind": "session-state",
            "session_id": session_id,
            "current_state": current_state,
            "slot_cache_keys": {name: slot.get("cache_key") for name, slot in slots.items()},
            "transition_count": transition_count,
            "states": machine.get("states", []),
            "initial_state": machine.get("initial_state"),
            "transition_schema": machine.get("transitions", []),
        }
    )


def _restore_prefix(
    previous_session: dict[str, Any] | None,
    start_turn: int | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, int]]:
    if previous_session is None or start_turn is None:
        return None, [], {"transition": 0, "rejected": 0, "change": 0, "failure": 0}
    checkpoints = previous_session.get("checkpoints", [])
    candidates = [checkpoint for checkpoint in checkpoints if checkpoint["turn"] < start_turn]
    if not candidates:
        return None, [], {"transition": 0, "rejected": 0, "change": 0, "failure": 0}
    checkpoint = candidates[-1]
    prefix = {
        "checkpoint": checkpoint,
        "transitions": previous_session.get("transition_trace", [])[: checkpoint["transition_count"]],
        "rejected": previous_session.get("rejected_transitions", [])[: checkpoint["rejected_count"]],
        "changes": previous_session.get("slot_changes", [])[: checkpoint["change_count"]],
        "failures": previous_session.get("failures", [])[: checkpoint["failure_count"]],
        "checkpoints": [item for item in checkpoints if item["turn"] <= checkpoint["turn"]],
    }
    return prefix, prefix["checkpoints"], {
        "transition": checkpoint["transition_count"],
        "rejected": checkpoint["rejected_count"],
        "change": checkpoint["change_count"],
        "failure": checkpoint["failure_count"],
    }


def evaluate_session(
    *,
    session_id: str,
    events: list[dict[str, Any]],
    machine: dict[str, Any],
    config: dict[str, Any],
    machine_hash: str,
    start_turn: int | None = None,
    previous_session: dict[str, Any] | None = None,
    dirty_slots: list[str] | None = None,
    pollution_reasons: list[str] | None = None,
    quota_override: int | None = None,
) -> dict[str, Any]:
    if not events:
        return {
            "session_id": session_id,
            "status": "complete",
            "current_state": machine["initial_state"],
            "slots": {},
            "transition_trace": [],
            "rejected_transitions": [],
            "slot_changes": [],
            "failures": [],
            "pending_cursor": None,
            "state_entries": 1,
            "quota": int(
                quota_override if quota_override is not None else config["session_state_quota"]
            ),
            "state_cache_key": _state_cache_key(
                session_id=session_id,
                current_state=machine["initial_state"],
                slots={},
                transition_count=0,
                machine=machine,
            ),
            "transition_trace_cache_key": stable_hash([]),
            "checkpoints": [],
            "committed": True,
            "reuse": {
                "checkpoint_turn": None,
                "replayed_events": 0,
                "dirty_slots": [],
                "pollution_reasons": pollution_reasons or [],
            },
        }

    prefix, checkpoints, counters = _restore_prefix(previous_session, start_turn)
    if prefix is None:
        current_state = machine["initial_state"]
        slots: dict[str, Any] = {}
        transition_trace: list[dict[str, Any]] = []
        rejected_transitions: list[dict[str, Any]] = []
        slot_changes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        checkpoint_turn: int | None = None
    else:
        checkpoint = prefix["checkpoint"]
        current_state = checkpoint["current_state"]
        slots = _clone_slots(checkpoint["slots"])
        transition_trace = list(prefix["transitions"])
        rejected_transitions = list(prefix["rejected"])
        slot_changes = list(prefix["changes"])
        failures = list(prefix["failures"])
        checkpoint_turn = checkpoint["turn"]

    dirty_slot_names = sorted(set(dirty_slots or []))
    replayed_events = 0
    quota = int(quota_override if quota_override is not None else config["session_state_quota"])
    status = "complete"
    pending_cursor: int | None = None

    for event in events:
        turn = int(event["turn"])
        if start_turn is not None and turn < start_turn:
            continue
        replayed_events += 1
        injection_id = event.get("_injection_id")
        event_failed = False

        staged_slots = _clone_slots(slots)
        staged_changes: list[dict[str, Any]] = []
        try:
            for update in event.get("slot_updates", []):
                slot_name = update["slot"]
                if slot_name not in machine["slots"]:
                    raise PipelineValidationError(f"unknown slot {slot_name!r}")
                slot_schema = machine["slots"][slot_name]
                strategy = machine["merge_strategies"][slot_name]
                next_slot, change = merge_slot_update(
                    old_state=staged_slots.get(slot_name),
                    update=update,
                    slot_name=slot_name,
                    slot_schema=slot_schema,
                    strategy=strategy,
                    machine=machine,
                    config=config,
                    event=event,
                    turn=turn,
                )
                staged_slots[slot_name] = next_slot
                staged_changes.append(change)
        except Exception as exc:  # A single event failure must not kill the session.
            failures.append(
                make_failure(
                    config,
                    "slot_value_invalid",
                    "merge",
                    str(exc),
                    session_id=session_id,
                    turn=turn,
                    exception_type=type(exc).__name__,
                    line=event.get("_line"),
                    injection_id=injection_id,
                )
            )
            event_failed = True

        if event_failed:
            checkpoint = {
                "turn": turn,
                "current_state": current_state,
                "slots": _clone_slots(slots),
                "transition_count": len(transition_trace),
                "rejected_count": len(rejected_transitions),
                "change_count": len(slot_changes),
                "failure_count": len(failures),
                "state_entries": state_entry_count(slots),
                "state_cache_key": _state_cache_key(
                    session_id=session_id,
                    current_state=current_state,
                    slots=slots,
                    transition_count=len(transition_trace),
                    machine=machine,
                ),
            }
            checkpoints.append(checkpoint)
            continue

        if state_entry_count(staged_slots) > quota:
            quota_injection_id = event.get("_quota_injection_id", injection_id)
            failures.append(
                make_failure(
                    config,
                    "session_quota_exceeded",
                    "merge",
                    f"session state quota {quota} would be exceeded at turn {turn}",
                    session_id=session_id,
                    turn=turn,
                    exception_type="QuotaExceeded",
                    injection_id=quota_injection_id,
                )
            )
            status = "quota_exceeded"
            pending_cursor = turn
            break

        slots = staged_slots
        slot_changes.extend(staged_changes)
        state_key = _state_cache_key(
            session_id=session_id,
            current_state=current_state,
            slots=slots,
            transition_count=len(transition_trace),
            machine=machine,
        )

        try:
            requested_target = event.get("requested_target", event.get("transition_target"))
            candidates = _candidate_rules(machine, current_state, slots, event)
            if requested_target is not None:
                candidates = [rule for rule in candidates if rule["to"] == requested_target]
                if candidates:
                    chosen = candidates[0]
                    source_state = current_state
                    current_state = chosen["to"]
                    transition_trace.append(
                        {
                            "from": source_state,
                            "to": current_state,
                            "turn": turn,
                            "rule_id": chosen["id"],
                            "candidate_rule_ids": [rule["id"] for rule in candidates],
                            "event_fingerprint": transition_event_hash(event),
                            "machine_parameter_fingerprint": stable_hash(chosen),
                            "cache_key": _transition_hash(
                                turn=turn,
                                source_state=source_state,
                                target_state=current_state,
                                rule_id=chosen["id"],
                                candidates=[rule["id"] for rule in candidates],
                                event=event,
                                machine=machine,
                                state_cache_key=state_key,
                            ),
                        }
                    )
                else:
                    rejected_transitions.append(
                        {
                            "from": current_state,
                            "to": requested_target,
                            "turn": turn,
                            "rule_id": None,
                            "code": business_code(config, "transition_rejected"),
                            "reason": "no_matching_rule_for_requested_target",
                            "candidate_rule_ids": [],
                            "cache_key": stable_hash(
                                {
                                    "kind": "rejected-transition",
                                    "turn": turn,
                                    "from": current_state,
                                    "to": requested_target,
                                    "event": transition_event_hash(event),
                                    "rule_schema": machine.get("transitions", []),
                                    "state_cache_key": state_key,
                                }
                            ),
                        }
                    )
            elif candidates:
                chosen = candidates[0]
                source_state = current_state
                current_state = chosen["to"]
                candidate_ids = [rule["id"] for rule in candidates]
                transition_trace.append(
                    {
                        "from": source_state,
                        "to": current_state,
                        "turn": turn,
                        "rule_id": chosen["id"],
                        "candidate_rule_ids": candidate_ids,
                        "event_fingerprint": transition_event_hash(event),
                        "machine_parameter_fingerprint": stable_hash(chosen),
                        "cache_key": _transition_hash(
                            turn=turn,
                            source_state=source_state,
                            target_state=current_state,
                            rule_id=chosen["id"],
                            candidates=candidate_ids,
                            event=event,
                            machine=machine,
                            state_cache_key=state_key,
                        ),
                    }
                )
        except Exception as exc:
            failures.append(
                make_failure(
                    config,
                    "internal_error",
                    "transition",
                    str(exc),
                    session_id=session_id,
                    turn=turn,
                    exception_type=type(exc).__name__,
                    line=event.get("_line"),
                    injection_id=injection_id,
                )
            )

        state_key = _state_cache_key(
            session_id=session_id,
            current_state=current_state,
            slots=slots,
            transition_count=len(transition_trace),
            machine=machine,
        )
        checkpoint = {
            "turn": turn,
            "current_state": current_state,
            "slots": _clone_slots(slots),
            "transition_count": len(transition_trace),
            "rejected_count": len(rejected_transitions),
            "change_count": len(slot_changes),
            "failure_count": len(failures),
            "state_entries": state_entry_count(slots),
            "state_cache_key": state_key,
        }
        checkpoints.append(checkpoint)

    state_key = _state_cache_key(
        session_id=session_id,
        current_state=current_state,
        slots=slots,
        transition_count=len(transition_trace),
        machine=machine,
    )
    entry_count = state_entry_count(slots)
    transition_trace_cache_key = stable_hash(
        [item.get("cache_key") for item in transition_trace]
    )
    return {
        "session_id": session_id,
        "status": status,
        "current_state": current_state,
        "slots": slots,
        "transition_trace": transition_trace,
        "rejected_transitions": rejected_transitions,
        "slot_changes": slot_changes,
        "failures": failures,
        "pending_cursor": pending_cursor,
        "state_entries": entry_count,
        "quota": quota,
        "state_cache_key": state_key,
        "transition_trace_cache_key": transition_trace_cache_key,
        "checkpoints": checkpoints,
        "committed": True,
        "reuse": {
            "checkpoint_turn": checkpoint_turn,
            "replayed_events": replayed_events,
            "dirty_slots": dirty_slot_names,
            "pollution_reasons": pollution_reasons or [],
        },
        "fingerprints": {
            "machine": machine_hash,
            "state": state_key,
        },
    }


def compare_state_projection(state: dict[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for session_id in sorted(state.get("sessions", {})):
        session = state["sessions"][session_id]
        projection[session_id] = {
            "status": session.get("status"),
            "current_state": session.get("current_state"),
            "slots": session.get("slots", {}),
            "transition_trace": session.get("transition_trace", []),
            "rejected_transitions": session.get("rejected_transitions", []),
            "slot_changes": session.get("slot_changes", []),
            "pending_cursor": session.get("pending_cursor"),
            "state_entries": session.get("state_entries"),
            "quota": session.get("quota"),
            "state_cache_key": session.get("state_cache_key"),
            "transition_trace_cache_key": session.get("transition_trace_cache_key"),
        }
    return projection

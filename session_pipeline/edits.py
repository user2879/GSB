from __future__ import annotations

import copy
from typing import Any

from .loading import make_failure
from .util import PipelineValidationError, stable_hash


def _event_slots(event: dict[str, Any]) -> list[str]:
    return sorted(
        {
            update.get("slot")
            for update in event.get("slot_updates", [])
            if isinstance(update, dict) and isinstance(update.get("slot"), str)
        }
    )


def _insert_event_by_timestamp(events: list[dict[str, Any]], event: dict[str, Any]) -> int:
    timestamp = event.get("timestamp_ms", 0)
    index = len(events)
    for position, existing in enumerate(events):
        if existing.get("timestamp_ms", 0) > timestamp:
            index = position
            break
    events.insert(index, event)
    return index


def _find_turn(events: list[dict[str, Any]], session_id: str, turn: int) -> int | None:
    for index, event in enumerate(events):
        if event.get("session_id") == session_id and event.get("turn") == turn:
            return index
    return None


def _record_change(change: dict[str, Any], session_id: str, turn: int, slot: str | None, reason: str) -> None:
    change["affected_sessions"].add(session_id)
    previous = change["session_turns"].get(session_id)
    change["session_turns"][session_id] = turn if previous is None else min(previous, turn)
    if slot:
        change["dirty_slots"].setdefault(session_id, set()).add(slot)


def apply_edit_batch(
    *,
    events: list[dict[str, Any]],
    machine: dict[str, Any],
    batch: dict[str, Any],
    config: dict[str, Any],
    batch_number: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    next_events = copy.deepcopy(events)
    next_machine = copy.deepcopy(machine)
    failures: list[dict[str, Any]] = []
    operations = batch.get("operations", [])
    if not isinstance(operations, list):
        return next_events, next_machine, {}, [
            make_failure(
                config,
                "input_invalid",
                "editing",
                f"batch {batch_number} operations must be a list",
            )
        ]

    event_change_count = sum(
        1
        for operation in operations
        if isinstance(operation, dict)
        and operation.get("op") in {"append", "replace_utterance", "replace_text", "delete_turn", "delete"}
    )
    if event_change_count > int(config["max_events_per_batch"]):
        failures.append(
            make_failure(
                config,
                "input_invalid",
                "editing",
                f"batch {batch_number} changes {event_change_count} events, above max_events_per_batch",
            )
        )
        return next_events, next_machine, {}, failures

    change = {
        "batch_number": batch_number,
        "batch_id": batch.get("batch_id", batch_number),
        "affected_sessions": set(),
        "session_turns": {},
        "dirty_slots": {},
        "reasons": [],
    }

    for operation_index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "editing",
                    f"batch {batch_number} operation #{operation_index} must be an object",
                )
            )
            continue
        operation_type = operation.get("op")
        try:
            if operation_type == "append":
                event = copy.deepcopy(operation.get("event"))
                if not isinstance(event, dict):
                    raise PipelineValidationError("append requires an event object")
                session_id = event.get("session_id")
                turn = event.get("turn")
                if not isinstance(session_id, str) or not session_id:
                    raise PipelineValidationError("appended event requires session_id")
                if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
                    raise PipelineValidationError("appended event requires a positive turn")
                _insert_event_by_timestamp(next_events, event)
                _record_change(change, session_id, turn, None, "append")
                for slot in _event_slots(event):
                    _record_change(change, session_id, turn, slot, "append")
                change["reasons"].append(
                    f"append:{session_id}:{turn}:{stable_hash(event)[:12]}"
                )
            elif operation_type in {"replace_utterance", "replace_text"}:
                session_id = operation.get("session_id")
                turn = operation.get("turn")
                text = operation.get("text")
                index = _find_turn(next_events, session_id, turn)
                if index is None:
                    raise PipelineValidationError(f"replace target not found: {session_id} turn {turn}")
                if not isinstance(text, str):
                    raise PipelineValidationError("replacement text must be a string")
                next_events[index]["text"] = text
                _record_change(change, session_id, turn, None, "replace_utterance")
                change["reasons"].append(f"replace_utterance:{session_id}:{turn}")
            elif operation_type in {"delete_turn", "delete"}:
                session_id = operation.get("session_id")
                turn = operation.get("turn")
                index = _find_turn(next_events, session_id, turn)
                if index is None:
                    raise PipelineValidationError(f"delete target not found: {session_id} turn {turn}")
                deleted = next_events.pop(index)
                slots = _event_slots(deleted)
                _record_change(change, session_id, turn, None, "delete_turn")
                for slot in slots:
                    _record_change(change, session_id, turn, slot, "delete_turn")
                for later in next_events:
                    if later.get("session_id") == session_id and later.get("turn", 0) > turn:
                        later["turn"] = int(later["turn"]) - 1
                        for slot in _event_slots(later):
                            _record_change(
                                change,
                                session_id,
                                turn,
                                slot,
                                "delete_turn_renumber",
                            )
                change["reasons"].append(f"delete_turn:{session_id}:{turn}")
            elif operation_type in {"patch_slot_schema", "slot_schema_change"}:
                slot_name = operation.get("slot")
                patch = operation.get("patch")
                if slot_name not in next_machine["slots"]:
                    raise PipelineValidationError(f"unknown slot schema {slot_name!r}")
                if not isinstance(patch, dict) or not patch:
                    raise PipelineValidationError("slot schema patch must be a non-empty object")
                next_machine["slots"][slot_name].update(copy.deepcopy(patch))
                for event in next_events:
                    if slot_name in _event_slots(event):
                        session_id = event.get("session_id")
                        turn = int(event.get("turn", 0))
                        _record_change(
                            change,
                            session_id,
                            turn,
                            slot_name,
                            "slot_schema_change",
                        )
                change["reasons"].append(f"slot_schema_change:{slot_name}:{stable_hash(patch)[:12]}")
            elif operation_type in {"set_merge_strategy", "merge_strategy_change"}:
                slot_name = operation.get("slot")
                strategy = operation.get("strategy", operation.get("value"))
                if slot_name not in next_machine["slots"]:
                    raise PipelineValidationError(f"unknown merge strategy slot {slot_name!r}")
                if not isinstance(strategy, str) or not strategy:
                    raise PipelineValidationError("merge strategy must be a non-empty string")
                next_machine["merge_strategies"][slot_name] = strategy
                for event in next_events:
                    if slot_name in _event_slots(event):
                        session_id = event.get("session_id")
                        _record_change(
                            change,
                            session_id,
                            int(event["turn"]),
                            slot_name,
                            "merge_strategy_change",
                        )
                change["reasons"].append(f"merge_strategy_change:{slot_name}:{strategy}")
            elif operation_type in {"set_conflict_priority", "conflict_priority_change"}:
                priority = operation.get("priority")
                if not isinstance(priority, list) or not all(isinstance(item, str) for item in priority):
                    raise PipelineValidationError("conflict priority must be a string list")
                next_machine["conflict_priority"] = list(priority)
                for event in next_events:
                    for slot in _event_slots(event):
                        _record_change(
                            change,
                            event["session_id"],
                            int(event["turn"]),
                            slot,
                            "conflict_priority_change",
                        )
                change["reasons"].append(f"conflict_priority_change:{priority}")
            else:
                raise PipelineValidationError(f"unsupported edit operation {operation_type!r}")
        except Exception as exc:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "editing",
                    str(exc),
                    session_id=operation.get("session_id"),
                    turn=operation.get("turn") if isinstance(operation.get("turn"), int) else None,
                    exception_type=type(exc).__name__,
                )
            )

    serializable_change = {
        "batch_number": batch_number,
        "batch_id": change["batch_id"],
        "affected_sessions": sorted(change["affected_sessions"]),
        "session_turns": {key: change["session_turns"][key] for key in sorted(change["session_turns"])},
        "dirty_slots": {
            key: sorted(change["dirty_slots"].get(key, set())) for key in sorted(change["dirty_slots"])
        },
        "reasons": change["reasons"],
    }
    return next_events, next_machine, serializable_change, failures


def materialize_final_edits(
    *,
    events: list[dict[str, Any]],
    machine: dict[str, Any],
    edit_plan: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current_events = copy.deepcopy(events)
    current_machine = copy.deepcopy(machine)
    failures: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for index, batch in enumerate(edit_plan.get("batches", []), start=1):
        if not isinstance(batch, dict):
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "editing",
                    f"edit batch #{index} must be an object",
                )
            )
            continue
        current_events, current_machine, change, batch_failures = apply_edit_batch(
            events=current_events,
            machine=current_machine,
            batch=batch,
            config=config,
            batch_number=int(batch.get("batch_id", index)),
        )
        failures.extend(batch_failures)
        batches.append(change)
    return current_events, current_machine, batches, failures

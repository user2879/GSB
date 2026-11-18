from __future__ import annotations

import copy
from typing import Any

from .util import stable_hash


def _find_event(events: list[dict[str, Any]], session_id: str, turn: int) -> int | None:
    for index, event in enumerate(events):
        if event.get("session_id") == session_id and event.get("turn") == turn:
            return index
    return None


def apply_batch_injections(
    *,
    batch_number: int,
    events: list[dict[str, Any]],
    injection_config: dict[str, Any],
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    next_events = copy.deepcopy(events)
    quota_overrides: dict[str, int] = {}
    hits: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    injections = injection_config.get("injections", []) if enabled else []
    for injection in injections:
        if not isinstance(injection, dict) or injection.get("batch") != batch_number:
            continue
        injection_id = injection.get("id")
        injection_type = injection.get("type")
        session_id = injection.get("session_id")
        turn = injection.get("turn")
        if not isinstance(injection_id, str) or not injection_id:
            misses.append({"batch": batch_number, "type": injection_type, "reason": "missing id"})
            continue
        if not isinstance(session_id, str) or not session_id:
            misses.append({"batch": batch_number, "injection_id": injection_id, "reason": "missing session"})
            continue
        if injection_type == "quota_overflow":
            quota = injection.get("quota")
            if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
                misses.append(
                    {
                        "batch": batch_number,
                        "injection_id": injection_id,
                        "reason": "quota must be a positive integer",
                    }
                )
                continue
            quota_overrides[session_id] = quota
            for event in next_events:
                if event.get("session_id") == session_id:
                    event["_quota_injection_id"] = injection_id
            hits.append(
                {
                    "injection_id": injection_id,
                    "type": injection_type,
                    "batch": batch_number,
                    "session_id": session_id,
                    "turn": turn,
                    "applied": {"quota": quota},
                }
            )
            continue

        if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
            misses.append(
                {"batch": batch_number, "injection_id": injection_id, "reason": "invalid target turn"}
            )
            continue
        index = _find_event(next_events, session_id, turn)
        if index is None:
            misses.append(
                {
                    "batch": batch_number,
                    "injection_id": injection_id,
                    "reason": f"target not found: {session_id} turn {turn}",
                }
            )
            continue

        if injection_type == "unknown_transition_target":
            target = injection.get("target", "INVALID_STATE")
            next_events[index] = copy.deepcopy(next_events[index])
            next_events[index]["requested_target"] = target
            next_events[index].pop("transition_target", None)
            next_events[index]["_injection_id"] = injection_id
            hits.append(
                {
                    "injection_id": injection_id,
                    "type": injection_type,
                    "batch": batch_number,
                    "session_id": session_id,
                    "turn": turn,
                    "applied": {"target": target},
                }
            )
        elif injection_type == "illegal_slot_value":
            slot_name = injection.get("slot")
            if not isinstance(slot_name, str) or not slot_name:
                misses.append(
                    {
                        "batch": batch_number,
                        "injection_id": injection_id,
                        "reason": "illegal_slot_value requires slot",
                    }
                )
                continue
            next_events[index] = copy.deepcopy(next_events[index])
            updates = next_events[index].setdefault("slot_updates", [])
            payload = {
                "slot": slot_name,
                "value": injection.get("value"),
                "confidence": injection.get("confidence", "0.500000"),
                "source": injection.get("source", "explicit"),
            }
            replaced = False
            for update_index, update in enumerate(updates):
                if update.get("slot") == slot_name:
                    updates[update_index] = payload
                    replaced = True
                    break
            if not replaced:
                updates.append(payload)
            next_events[index]["_injection_id"] = injection_id
            hits.append(
                {
                    "injection_id": injection_id,
                    "type": injection_type,
                    "batch": batch_number,
                    "session_id": session_id,
                    "turn": turn,
                    "applied": {"slot": slot_name, "value": injection.get("value")},
                }
            )
        elif injection_type == "turn_jump":
            offset = injection.get("offset", 2)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
                misses.append(
                    {
                        "batch": batch_number,
                        "injection_id": injection_id,
                        "reason": "turn_jump offset must be a positive integer",
                    }
                )
                continue
            jumped = copy.deepcopy(next_events[index])
            jumped["turn"] = int(jumped["turn"]) + offset
            maximum_timestamp = max((int(event.get("timestamp_ms", 0)) for event in next_events), default=0)
            jumped["timestamp_ms"] = maximum_timestamp + 1 + batch_number
            jumped["_injection_id"] = injection_id
            jumped["_injection_fingerprint"] = stable_hash(jumped)
            next_events.append(jumped)
            hits.append(
                {
                    "injection_id": injection_id,
                    "type": injection_type,
                    "batch": batch_number,
                    "session_id": session_id,
                    "turn": turn,
                    "applied": {"injected_turn": jumped["turn"]},
                }
            )
        else:
            misses.append(
                {
                    "batch": batch_number,
                    "injection_id": injection_id,
                    "reason": f"unsupported injection type {injection_type!r}",
                }
            )
    return next_events, quota_overrides, hits, misses


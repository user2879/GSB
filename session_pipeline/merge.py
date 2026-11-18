from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from .util import (
    PipelineValidationError,
    decimal_string,
    fixed_to_string,
    parse_decimal,
    quantize_decimal,
    slot_event_hash,
    stable_hash,
)


def parse_slot_value(slot_schema: dict[str, Any], raw_value: Any) -> tuple[Any, Any]:
    """Return (canonical JSON value, comparable internal value)."""
    slot_type = slot_schema.get("type")
    if slot_type == "string":
        if not isinstance(raw_value, str):
            raise PipelineValidationError("string slot value must be a string")
        pattern = slot_schema.get("pattern")
        if pattern is not None and re.fullmatch(str(pattern), raw_value) is None:
            raise PipelineValidationError(f"string slot value does not match {pattern!r}")
        if "min_length" in slot_schema and len(raw_value) < int(slot_schema["min_length"]):
            raise PipelineValidationError("string slot value is too short")
        if "max_length" in slot_schema and len(raw_value) > int(slot_schema["max_length"]):
            raise PipelineValidationError("string slot value is too long")
        return raw_value, raw_value
    if slot_type == "int":
        if isinstance(raw_value, bool):
            raise PipelineValidationError("int slot value must not be a boolean")
        if isinstance(raw_value, int):
            parsed = raw_value
        elif isinstance(raw_value, str) and re.fullmatch(r"[+-]?\d+", raw_value.strip()):
            parsed = int(raw_value.strip())
        else:
            raise PipelineValidationError("int slot value must be an integer")
        if "min" in slot_schema and parsed < int(slot_schema["min"]):
            raise PipelineValidationError(f"int slot value is below minimum {slot_schema['min']}")
        if "max" in slot_schema and parsed > int(slot_schema["max"]):
            raise PipelineValidationError(f"int slot value is above maximum {slot_schema['max']}")
        return parsed, parsed
    if slot_type == "bool":
        if isinstance(raw_value, bool):
            parsed_bool = raw_value
        elif isinstance(raw_value, str) and raw_value.strip().lower() in {"true", "false"}:
            parsed_bool = raw_value.strip().lower() == "true"
        else:
            raise PipelineValidationError("bool slot value must be true or false")
        return parsed_bool, parsed_bool
    if slot_type == "decimal":
        parsed_decimal = parse_decimal(raw_value, "decimal slot value")
        if "min" in slot_schema and parsed_decimal < parse_decimal(slot_schema["min"], "min"):
            raise PipelineValidationError(f"decimal slot value is below minimum {slot_schema['min']}")
        if "max" in slot_schema and parsed_decimal > parse_decimal(slot_schema["max"], "max"):
            raise PipelineValidationError(f"decimal slot value is above maximum {slot_schema['max']}")
        places = int(slot_schema.get("decimal_places", 4))
        parsed_decimal = quantize_decimal(parsed_decimal, places)
        return decimal_string(parsed_decimal, places), parsed_decimal
    if slot_type == "enum":
        allowed = slot_schema.get("allowed", [])
        for candidate in allowed:
            if raw_value == candidate:
                return candidate, candidate
        raise PipelineValidationError(f"enum slot value {raw_value!r} is not allowed")
    raise PipelineValidationError(f"unsupported slot type {slot_type!r}")


def parse_confidence(raw_value: Any, decimals: int) -> int:
    parsed = parse_decimal(1 if raw_value is None else raw_value, "confidence")
    if parsed < 0 or parsed > 1:
        raise PipelineValidationError("confidence must be in [0, 1]")
    return int(quantize_decimal(parsed, decimals) * (10**decimals))


def _priority_rank(source: str, priority: list[str]) -> int:
    try:
        return priority.index(source)
    except ValueError:
        return len(priority) + 1


def _priority_decision(
    old_source: str,
    new_source: str,
    priority: list[str],
    old_turn: int,
    new_turn: int,
) -> tuple[str, str]:
    old_rank = _priority_rank(old_source, priority)
    new_rank = _priority_rank(new_source, priority)
    if new_rank < old_rank:
        return "incoming", f"source priority {new_source!r} wins over {old_source!r}"
    if old_rank < new_rank:
        return "existing", f"source priority {old_source!r} wins over {new_source!r}"
    if new_turn > old_turn:
        return "incoming", "equal source priority; later turn wins"
    return "existing", "equal source priority and turn; existing value retained"


def _slot_hash_context(
    machine: dict[str, Any],
    slot_name: str,
    slot_schema: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    return {
        "slot": slot_name,
        "slot_schema": slot_schema,
        "merge_strategy": strategy,
        "conflict_priority": machine.get("conflict_priority", []),
    }


def merge_slot_update(
    *,
    old_state: dict[str, Any] | None,
    update: dict[str, Any],
    slot_name: str,
    slot_schema: dict[str, Any],
    strategy: str,
    machine: dict[str, Any],
    config: dict[str, Any],
    event: dict[str, Any],
    turn: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    confidence_decimals = int(config["confidence_decimals"])
    scale = 10**confidence_decimals
    incoming_value, incoming_comparable = parse_slot_value(slot_schema, update["value"])
    incoming_confidence = parse_confidence(update.get("confidence", "1.0"), confidence_decimals)
    source = update.get("source", "explicit")
    if not isinstance(source, str) or not source:
        source = "explicit"
    event_hash = slot_event_hash(event, update)
    parameter_hash = stable_hash(_slot_hash_context(machine, slot_name, slot_schema, strategy))
    base_update_hash = stable_hash(
        {
            "event": event_hash,
            "value": incoming_value,
            "confidence_fixed": incoming_confidence,
            "source": source,
            "parameter": parameter_hash,
        }
    )

    before = None
    if old_state is not None:
        before = {
            "value": old_state["value"],
            "confidence_fixed": old_state["confidence_fixed"],
            "confidence": old_state["confidence"],
        }

    accepted = True
    decision_basis = "slot was empty"
    next_value = incoming_value
    next_confidence = incoming_confidence
    previous_hash = old_state.get("history_hash", "") if old_state else ""

    old_comparable = None
    if old_state is not None:
        _, old_comparable = parse_slot_value(slot_schema, old_state["value"])
        old_confidence = int(old_state["confidence_fixed"])
        same_value = old_comparable == incoming_comparable
        same_confidence = old_confidence == incoming_confidence
        priority = list(machine.get("conflict_priority", []))

        if strategy == "reject_overwrite":
            accepted = False
            next_value = old_state["value"]
            next_confidence = old_confidence
            decision_basis = "reject_overwrite: existing slot value is immutable"
        elif strategy == "overwrite":
            if same_value or same_confidence:
                winner, basis = _priority_decision(
                    old_state.get("source", "explicit"),
                    source,
                    priority,
                    int(old_state.get("updated_turn", 0)),
                    turn,
                )
                if winner == "existing":
                    accepted = False
                    next_value = old_state["value"]
                    next_confidence = old_confidence
                decision_basis = basis
            else:
                decision_basis = "overwrite accepts the newest value"
        elif strategy == "max_value":
            if same_value:
                if old_confidence > incoming_confidence:
                    accepted = False
                    next_value = old_state["value"]
                    next_confidence = old_confidence
                    decision_basis = "equal value; existing confidence is higher"
                elif incoming_confidence > old_confidence:
                    decision_basis = "equal value; incoming confidence is higher"
                else:
                    winner, basis = _priority_decision(
                        old_state.get("source", "explicit"),
                        source,
                        priority,
                        int(old_state.get("updated_turn", 0)),
                        turn,
                    )
                    if winner == "existing":
                        accepted = False
                        next_value = old_state["value"]
                        next_confidence = old_confidence
                    decision_basis = basis
            elif same_confidence:
                winner, basis = _priority_decision(
                    old_state.get("source", "explicit"),
                    source,
                    priority,
                    int(old_state.get("updated_turn", 0)),
                    turn,
                )
                if winner == "existing":
                    accepted = False
                    next_value = old_state["value"]
                    next_confidence = old_confidence
                decision_basis = basis + "; equal confidence tie"
            elif incoming_comparable > old_comparable:
                decision_basis = "max_value accepts the larger incoming value"
            else:
                accepted = False
                next_value = old_state["value"]
                next_confidence = old_confidence
                decision_basis = "max_value retains the larger existing value"
        elif strategy == "weighted":
            if isinstance(incoming_comparable, bool) or not isinstance(
                incoming_comparable, (int, Decimal)
            ) or isinstance(old_comparable, bool) or not isinstance(old_comparable, (int, Decimal)):
                raise PipelineValidationError("weighted merge requires numeric slot values")
            old_fixed = int(old_state["confidence_fixed"])
            total_weight = old_fixed + incoming_confidence
            if total_weight == 0:
                winner, basis = _priority_decision(
                    old_state.get("source", "explicit"),
                    source,
                    priority,
                    int(old_state.get("updated_turn", 0)),
                    turn,
                )
                if winner == "existing":
                    accepted = False
                    next_value = old_state["value"]
                    next_confidence = old_fixed
                decision_basis = "zero total confidence; " + basis
            else:
                old_number = Decimal(old_comparable)
                incoming_number = Decimal(incoming_comparable)
                weighted_number = (
                    old_number * Decimal(old_fixed) + incoming_number * Decimal(incoming_confidence)
                ) / Decimal(total_weight)
                if slot_schema.get("type") == "int":
                    next_value = int(quantize_decimal(weighted_number, 0))
                else:
                    places = int(slot_schema.get("decimal_places", 4))
                    next_value = decimal_string(quantize_decimal(weighted_number, places), places)
                next_confidence = old_fixed + (
                    (scale - old_fixed) * incoming_confidence + scale // 2
                ) // scale
                next_confidence = min(scale, next_confidence)
                decision_basis = "confidence-weighted numeric accumulation"
        else:
            raise PipelineValidationError(f"unsupported merge strategy {strategy!r}")

    if accepted:
        next_state = {
            "value": next_value,
            "confidence_fixed": int(next_confidence),
            "confidence": fixed_to_string(int(next_confidence), confidence_decimals),
            "merge_strategy": strategy,
            "source": source,
            "updated_turn": turn,
            "event_fingerprint": event_hash,
            "parameter_fingerprint": parameter_hash,
            "history_hash": stable_hash(
                {"previous": previous_hash, "update": base_update_hash, "accepted": True}
            ),
        }
    else:
        next_state = dict(old_state or {})
        next_state["history_hash"] = stable_hash(
            {"previous": previous_hash, "update": base_update_hash, "accepted": False}
        )

    change_cache_key = stable_hash(
        {
            "kind": "slot-change",
            "update": base_update_hash,
            "before": before,
            "after": {
                "value": next_state.get("value"),
                "confidence_fixed": next_state.get("confidence_fixed"),
            },
            "accepted": accepted,
            "decision": decision_basis,
        }
    )
    aggregate_cache_key = stable_hash(
        {"previous": previous_hash, "change_cache_key": change_cache_key, "accepted": accepted}
    )
    next_state["cache_key"] = aggregate_cache_key
    next_state["history_hash"] = aggregate_cache_key
    next_state["last_change_cache_key"] = change_cache_key
    change = {
        "turn": turn,
        "slot": slot_name,
        "strategy": strategy,
        "source": source,
        "accepted": accepted,
        "before": before,
        "after": {
            "value": next_state.get("value"),
            "confidence_fixed": next_state.get("confidence_fixed"),
            "confidence": next_state.get("confidence"),
        },
        "decision_basis": decision_basis,
        "cache_key": change_cache_key,
        "event_fingerprint": event_hash,
        "parameter_fingerprint": parameter_hash,
    }
    return next_state, change

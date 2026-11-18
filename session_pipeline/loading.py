from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .util import (
    DEFAULT_BUSINESS_CODES,
    PipelineValidationError,
    business_code,
    canonical_json,
    parse_decimal,
    read_json,
    read_jsonl_raw,
    stable_hash,
)


SUPPORTED_SLOT_TYPES = {"string", "int", "bool", "decimal", "enum"}
SUPPORTED_MERGE_STRATEGIES = {"overwrite", "max_value", "weighted", "reject_overwrite"}
SUPPORTED_RULE_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "exists", "not_exists", "in"}


def make_failure(
    config: dict[str, Any],
    code_key: str,
    phase: str,
    message: str,
    *,
    session_id: str | None = None,
    turn: int | None = None,
    exception_type: str = "ValidationError",
    line: int | None = None,
    injection_id: str | None = None,
) -> dict[str, Any]:
    failure = {
        "code": business_code(config, code_key),
        "phase": phase,
        "session_id": session_id,
        "turn": turn,
        "exception_type": exception_type,
        "message": message,
    }
    if line is not None:
        failure["line"] = line
    if injection_id is not None:
        failure["injection_id"] = injection_id
    return failure


def normalize_config(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    config = raw if isinstance(raw, dict) else {}
    if not isinstance(raw, dict):
        failures.append(
            make_failure(
                {"business_codes": DEFAULT_BUSINESS_CODES},
                "input_invalid",
                "loading",
                "pipeline config must be a JSON object",
            )
        )

    defaults = {
        "worker_count": 2,
        "max_events_per_batch": 64,
        "global_state_budget": 64,
        "session_state_quota": 16,
        "snapshot_interval_batches": 1,
        "confidence_decimals": 6,
        "conflict_priority": ["explicit", "inferred", "default"],
        "merge_strategy_overrides": {},
    }
    normalized = dict(defaults)
    normalized.update(config)

    integer_rules = {
        "worker_count": (1, 1024),
        "max_events_per_batch": (1, 1_000_000),
        "global_state_budget": (0, 1_000_000_000),
        "session_state_quota": (1, 1_000_000_000),
        "snapshot_interval_batches": (1, 1_000_000),
        "confidence_decimals": (0, 9),
    }
    for key, (minimum, maximum) in integer_rules.items():
        value = normalized.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            failures.append(
                make_failure(
                    normalized,
                    "input_invalid",
                    "loading",
                    f"config.{key} must be an integer in [{minimum}, {maximum}]",
                )
            )
            normalized[key] = defaults.get(key, minimum)

    codes = normalized.get("business_codes")
    if not isinstance(codes, dict):
        codes = {}
    normalized["business_codes"] = {**DEFAULT_BUSINESS_CODES, **codes}
    for key, value in normalized["business_codes"].items():
        if not isinstance(value, str) or not value:
            normalized["business_codes"][key] = DEFAULT_BUSINESS_CODES.get(key, "E_INPUT_INVALID")
            failures.append(
                make_failure(
                    normalized,
                    "input_invalid",
                    "loading",
                    f"config.business_codes.{key} must be a non-empty string",
                )
            )

    priority = normalized.get("conflict_priority")
    if not isinstance(priority, list) or not priority or not all(
        isinstance(item, str) and item for item in priority
    ):
        failures.append(
            make_failure(
                normalized,
                "input_invalid",
                "loading",
                "config.conflict_priority must be a non-empty string list",
            )
        )
        normalized["conflict_priority"] = list(defaults["conflict_priority"])
    elif len(set(priority)) != len(priority):
        failures.append(
            make_failure(
                normalized,
                "input_invalid",
                "loading",
                "config.conflict_priority must not contain duplicates",
            )
        )

    overrides = normalized.get("merge_strategy_overrides")
    if not isinstance(overrides, dict):
        failures.append(
            make_failure(
                normalized,
                "input_invalid",
                "loading",
                "config.merge_strategy_overrides must be an object",
            )
        )
        normalized["merge_strategy_overrides"] = {}
    else:
        for slot, strategy in overrides.items():
            if strategy not in SUPPORTED_MERGE_STRATEGIES:
                failures.append(
                    make_failure(
                        normalized,
                        "input_invalid",
                        "loading",
                        f"unsupported merge strategy override for {slot}: {strategy!r}",
                    )
                )
    return normalized, failures


def normalize_machine(
    raw: Any,
    config: dict[str, Any],
    *,
    apply_config_overrides: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        failures.append(
            make_failure(config, "input_invalid", "loading", "state machine YAML must be an object")
        )
        raw = {}

    states_raw = raw.get("states")
    states: list[str] = []
    if not isinstance(states_raw, list) or not states_raw:
        failures.append(
            make_failure(config, "input_invalid", "loading", "state machine states must be non-empty")
        )
    else:
        for state in states_raw:
            if not isinstance(state, str) or not state:
                failures.append(
                    make_failure(config, "input_invalid", "loading", "state names must be non-empty strings")
                )
            elif state in states:
                failures.append(
                    make_failure(config, "input_invalid", "loading", f"duplicate state: {state}")
                )
            else:
                states.append(state)

    initial_state = raw.get("initial_state")
    if not isinstance(initial_state, str) or initial_state not in states:
        failures.append(
            make_failure(
                config,
                "input_invalid",
                "loading",
                "initial_state must name exactly one declared state",
            )
        )
        initial_state = states[0] if states else ""

    raw_slots = raw.get("slots")
    slots: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_slots, dict) or not raw_slots:
        failures.append(
            make_failure(config, "input_invalid", "loading", "slot schema must be a non-empty object")
        )
    else:
        for slot_name, slot_schema in raw_slots.items():
            if not isinstance(slot_name, str) or not slot_name:
                failures.append(
                    make_failure(config, "input_invalid", "loading", "slot names must be non-empty strings")
                )
                continue
            if not isinstance(slot_schema, dict):
                failures.append(
                    make_failure(config, "input_invalid", "loading", f"slot {slot_name} schema must be an object")
                )
                continue
            slot_type = slot_schema.get("type")
            if slot_type not in SUPPORTED_SLOT_TYPES:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        f"slot {slot_name} uses unsupported type {slot_type!r}",
                    )
                )
                continue
            normalized_slot = dict(slot_schema)
            if slot_type == "enum":
                allowed = normalized_slot.get("allowed")
                if not isinstance(allowed, list) or not allowed:
                    failures.append(
                        make_failure(
                            config,
                            "input_invalid",
                            "loading",
                            f"enum slot {slot_name} requires a non-empty allowed list",
                        )
                    )
                    continue
            elif slot_type == "int":
                for bound in ("min", "max"):
                    if bound in normalized_slot and (
                        isinstance(normalized_slot[bound], bool)
                        or not isinstance(normalized_slot[bound], int)
                    ):
                        failures.append(
                            make_failure(
                                config,
                                "input_invalid",
                                "loading",
                                f"int slot {slot_name}.{bound} must be an integer",
                            )
                        )
            elif slot_type == "decimal":
                try:
                    for bound in ("min", "max"):
                        if bound in normalized_slot:
                            normalized_slot[bound] = str(parse_decimal(normalized_slot[bound], bound))
                    if "decimal_places" in normalized_slot and (
                        isinstance(normalized_slot["decimal_places"], bool)
                        or not isinstance(normalized_slot["decimal_places"], int)
                        or not 0 <= normalized_slot["decimal_places"] <= 9
                    ):
                        raise PipelineValidationError("decimal_places must be in [0, 9]")
                except PipelineValidationError as exc:
                    failures.append(
                        make_failure(config, "input_invalid", "loading", f"slot {slot_name}: {exc}")
                    )
            elif slot_type == "string" and "pattern" in normalized_slot:
                try:
                    re.compile(str(normalized_slot["pattern"]))
                except re.error as exc:
                    failures.append(
                        make_failure(
                            config,
                            "input_invalid",
                            "loading",
                            f"slot {slot_name} has an invalid pattern: {exc}",
                        )
                    )
                    continue
            slots[slot_name] = normalized_slot

    raw_strategies = raw.get("merge_strategies", {})
    strategies: dict[str, str] = {}
    if raw_strategies is None:
        raw_strategies = {}
    if not isinstance(raw_strategies, dict):
        failures.append(
            make_failure(config, "input_invalid", "loading", "merge_strategies must be an object")
        )
        raw_strategies = {}
    for slot_name, strategy in raw_strategies.items():
        if slot_name not in slots:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"merge strategy references unknown slot {slot_name!r}",
                )
            )
            continue
        if strategy not in SUPPORTED_MERGE_STRATEGIES:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"slot {slot_name} uses unsupported merge strategy {strategy!r}",
                )
            )
            continue
        strategies[slot_name] = strategy

    for slot_name, slot_schema in list(slots.items()):
        strategy = strategies.get(slot_name) or slot_schema.get("merge")
        if strategy not in SUPPORTED_MERGE_STRATEGIES:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"slot {slot_name} has no usable merge strategy",
                )
            )
            slots.pop(slot_name, None)
            continue
        strategies[slot_name] = strategy
    if apply_config_overrides:
        for slot_name, strategy in config.get("merge_strategy_overrides", {}).items():
            if slot_name in slots and strategy in SUPPORTED_MERGE_STRATEGIES:
                strategies[slot_name] = strategy
            elif slot_name not in slots:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        f"merge strategy override references unknown slot {slot_name!r}",
                    )
                )

    priority = raw.get("conflict_priority", config.get("conflict_priority"))
    if not isinstance(priority, list) or not priority or not all(
        isinstance(item, str) and item for item in priority
    ):
        failures.append(
            make_failure(config, "input_invalid", "loading", "conflict_priority must be a string list")
        )
        priority = list(config.get("conflict_priority", ["explicit", "inferred", "default"]))

    rules_raw = raw.get("transitions", [])
    transitions: list[dict[str, Any]] = []
    seen_rule_ids: set[str] = set()
    if not isinstance(rules_raw, list):
        failures.append(
            make_failure(config, "input_invalid", "loading", "transitions must be a list")
        )
        rules_raw = []
    for index, rule in enumerate(rules_raw):
        if not isinstance(rule, dict):
            failures.append(
                make_failure(config, "input_invalid", "loading", f"transition #{index} must be an object")
            )
            continue
        rule_id = rule.get("id")
        source = rule.get("from")
        target = rule.get("to")
        if not isinstance(rule_id, str) or not rule_id:
            failures.append(
                make_failure(config, "input_invalid", "loading", f"transition #{index} has no id")
            )
            continue
        if rule_id in seen_rule_ids:
            failures.append(
                make_failure(config, "input_invalid", "loading", f"duplicate transition id {rule_id}")
            )
            continue
        if source != "*" and source not in states:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"transition {rule_id} references unknown source state {source!r}",
                )
            )
            continue
        if target not in states:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"transition {rule_id} references unknown target state {target!r}",
                )
            )
            continue
        conditions = rule.get("when", {})
        if conditions is None:
            conditions = {}
        if not isinstance(conditions, dict):
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"transition {rule_id} conditions must be an object",
                )
            )
            continue
        valid_conditions = True
        for slot_name, condition in conditions.items():
            if slot_name not in slots:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        f"transition {rule_id} references unknown slot {slot_name!r}",
                    )
                )
                valid_conditions = False
                break
            if not isinstance(condition, dict) or condition.get("op") not in SUPPORTED_RULE_OPS:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        f"transition {rule_id} has invalid condition for {slot_name}",
                    )
                )
                valid_conditions = False
                break
        text_contains = rule.get("text_contains")
        if text_contains is not None and not isinstance(text_contains, str):
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"transition {rule_id}.text_contains must be a string",
                )
            )
            valid_conditions = False
        roles = rule.get("roles")
        if roles is not None and (
            not isinstance(roles, list) or not all(isinstance(role, str) for role in roles)
        ):
            failures.append(
                make_failure(config, "input_invalid", "loading", f"transition {rule_id}.roles must be a list")
            )
            valid_conditions = False
        if not valid_conditions:
            continue
        seen_rule_ids.add(rule_id)
        normalized_rule = dict(rule)
        normalized_rule["when"] = conditions
        transitions.append(normalized_rule)

    transitions.sort(key=lambda item: item["id"])
    machine = {
        "machine_id": raw.get("machine_id", "state_machine"),
        "initial_state": initial_state,
        "states": states,
        "slots": slots,
        "merge_strategies": strategies,
        "conflict_priority": priority,
        "transitions": transitions,
    }
    if not states or not slots or not initial_state:
        failures.append(
            make_failure(config, "input_invalid", "loading", "state machine is not runnable")
        )
    return machine, failures


def machine_fingerprint(machine: dict[str, Any]) -> str:
    return stable_hash(machine)


def config_fingerprint(config: dict[str, Any]) -> str:
    return stable_hash(config)


def validate_events(
    raw_events: list[dict[str, Any]], machine: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    expected_turn: dict[str, int] = {}
    last_timestamp: int | None = None
    states = set(machine.get("states", []))

    for raw_event in raw_events:
        event = dict(raw_event)
        line = event.get("_line")
        session_id = event.get("session_id")
        turn = event.get("turn")
        injection_id = event.get("_injection_id")
        if not isinstance(session_id, str) or not session_id.strip():
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    "session_id must be a non-empty string",
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        if isinstance(turn, bool) or not isinstance(turn, int) or turn <= 0:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    "turn must be a positive integer",
                    session_id=session_id,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        expected = expected_turn.get(session_id, 1)
        if turn != expected:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"turn sequence for session {session_id} expected {expected}, got {turn}",
                    session_id=session_id,
                    turn=turn,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        role = event.get("role")
        if not isinstance(role, str) or not role:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    "role must be a non-empty string",
                    session_id=session_id,
                    turn=turn,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        text = event.get("text", "")
        if not isinstance(text, str):
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    "text must be a string",
                    session_id=session_id,
                    turn=turn,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        timestamp = event.get("timestamp_ms")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    "timestamp_ms must be a non-negative integer",
                    session_id=session_id,
                    turn=turn,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        if last_timestamp is not None and timestamp < last_timestamp:
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    f"timestamp_ms is not monotonic: {timestamp} < {last_timestamp}",
                    session_id=session_id,
                    turn=turn,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        updates = event.get("slot_updates", [])
        if updates is None:
            updates = []
        if not isinstance(updates, list) or any(not isinstance(update, dict) for update in updates):
            failures.append(
                make_failure(
                    config,
                    "input_invalid",
                    "loading",
                    "slot_updates must be a list of objects",
                    session_id=session_id,
                    turn=turn,
                    line=line,
                    injection_id=injection_id,
                )
            )
            continue
        invalid_update = False
        for update in updates:
            if not isinstance(update.get("slot"), str) or not update["slot"]:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        "slot update requires a non-empty slot name",
                        session_id=session_id,
                        turn=turn,
                        line=line,
                        injection_id=injection_id,
                    )
                )
                invalid_update = True
                break
            if "value" not in update:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        f"slot update for {update['slot']} has no value",
                        session_id=session_id,
                        turn=turn,
                        line=line,
                        injection_id=injection_id,
                    )
                )
                invalid_update = True
                break
        if invalid_update:
            continue
        requested_target = event.get("requested_target", event.get("transition_target"))
        if requested_target is not None:
            if not isinstance(requested_target, str) or not requested_target:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        "requested_target must be a non-empty string",
                        session_id=session_id,
                        turn=turn,
                        line=line,
                        injection_id=injection_id,
                    )
                )
                continue
            if requested_target not in states:
                failures.append(
                    make_failure(
                        config,
                        "input_invalid",
                        "loading",
                        f"unknown transition target {requested_target!r}",
                        session_id=session_id,
                        turn=turn,
                        line=line,
                        injection_id=injection_id,
                    )
                )
                continue

        event["slot_updates"] = updates
        event["text"] = text
        event["timestamp_ms"] = timestamp
        accepted.append(event)
        expected_turn[session_id] = turn + 1
        last_timestamp = timestamp
    return accepted, failures


def load_scenario_base(scenario_dir: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    config_raw: Any = {}
    try:
        config_raw = read_json(scenario_dir / "pipeline_config.json")
    except (OSError, ValueError) as exc:
        failures.append(
            make_failure(
                {"business_codes": DEFAULT_BUSINESS_CODES},
                "input_invalid",
                "loading",
                f"cannot read pipeline_config.json: {exc}",
            )
        )
    config, config_failures = normalize_config(config_raw)
    failures.extend(config_failures)

    machine_raw: Any = {}
    try:
        with (scenario_dir / "state_machine.yaml").open("r", encoding="utf-8") as handle:
            machine_raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        failures.append(
            make_failure(config, "input_invalid", "loading", f"cannot read state_machine.yaml: {exc}")
        )
    machine, machine_failures = normalize_machine(machine_raw, config)
    failures.extend(machine_failures)

    raw_events: list[dict[str, Any]] = []
    try:
        raw_events, jsonl_failures = read_jsonl_raw(scenario_dir / "events.jsonl")
        for failure in jsonl_failures:
            failure["code"] = business_code(config, "input_invalid")
        failures.extend(jsonl_failures)
    except OSError as exc:
        failures.append(
            make_failure(config, "input_invalid", "loading", f"cannot read events.jsonl: {exc}")
        )

    edit_plan: dict[str, Any] = {"batches": []}
    try:
        edit_plan = read_json(scenario_dir / "edit_plan.json")
        if not isinstance(edit_plan, dict) or not isinstance(edit_plan.get("batches", []), list):
            raise ValueError("edit plan must contain a batches list")
    except (OSError, ValueError) as exc:
        failures.append(
            make_failure(config, "input_invalid", "loading", f"cannot read edit_plan.json: {exc}")
        )

    injection_config: dict[str, Any] = {"injections": []}
    try:
        injection_config = read_json(scenario_dir / "fault_injections.json")
        if not isinstance(injection_config, dict) or not isinstance(
            injection_config.get("injections", []), list
        ):
            raise ValueError("fault injection config must contain an injections list")
    except (OSError, ValueError) as exc:
        failures.append(
            make_failure(config, "input_invalid", "loading", f"cannot read fault_injections.json: {exc}")
        )

    return {
        "raw_events": raw_events,
        "machine": machine,
        "config": config,
        "edit_plan": edit_plan,
        "injection_config": injection_config,
        "failures": failures,
        "machine_hash": machine_fingerprint(machine),
        "config_hash": config_fingerprint(config),
    }


def debug_input_summary(base: dict[str, Any]) -> str:
    return canonical_json(
        {
            "events": len(base["raw_events"]),
            "states": len(base["machine"]["states"]),
            "slots": len(base["machine"]["slots"]),
            "rules": len(base["machine"]["transitions"]),
        }
    )

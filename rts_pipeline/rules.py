"""Faction rule validation and initial authoritative state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import hash_value
from .constants import SCHEMA_VERSION
from .errors import PipelineError, require


def validate_rules(rules: Any) -> dict[str, Any]:
    require(isinstance(rules, dict), "E_SCHEMA", "rules must be a JSON object")
    require(
        rules.get("schema_version") == SCHEMA_VERSION,
        "E_SCHEMA",
        "unsupported rules schema version",
        expected=SCHEMA_VERSION,
        actual=rules.get("schema_version"),
    )
    factions = rules.get("factions")
    require(isinstance(factions, dict) and bool(factions), "E_SCHEMA", "factions must be non-empty")
    normalized: dict[str, Any] = {}
    for faction in sorted(factions):
        body = factions[faction]
        require(isinstance(body, dict), "E_SCHEMA", "faction rule must be an object", faction=faction)
        for field in ("resource_cap", "collect_rate", "queue_capacity"):
            value = body.get(field)
            require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                "E_SCHEMA",
                "rule field must be a positive integer",
                faction=faction,
                field=field,
                actual=value,
            )
        costs = body.get("unit_costs")
        build_ticks = body.get("build_ticks")
        require(isinstance(costs, dict) and bool(costs), "E_SCHEMA", "unit_costs must be non-empty", faction=faction)
        require(isinstance(build_ticks, dict), "E_SCHEMA", "build_ticks must be an object", faction=faction)
        for unit in sorted(costs):
            cost = costs[unit]
            build = build_ticks.get(unit)
            require(isinstance(cost, int) and not isinstance(cost, bool) and cost > 0, "E_SCHEMA", "unit cost must be positive", faction=faction, unit=unit)
            require(isinstance(build, int) and not isinstance(build, bool) and build >= 0, "E_SCHEMA", "build tick must be non-negative", faction=faction, unit=unit)
        normalized[faction] = {
            "resource_cap": body["resource_cap"],
            "collect_rate": body["collect_rate"],
            "queue_capacity": body["queue_capacity"],
            "unit_costs": {unit: costs[unit] for unit in sorted(costs)},
            "build_ticks": {unit: build_ticks[unit] for unit in sorted(costs)},
        }
    return {"schema_version": SCHEMA_VERSION, "factions": normalized}


def rules_hash(rules: dict[str, Any]) -> str:
    return hash_value(validate_rules(rules))


def initial_state(rules: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_rules(rules)
    factions: dict[str, Any] = {}
    cursors: dict[str, int] = {}
    for faction in sorted(normalized["factions"]):
        factions[faction] = {
            "resources": 0,
            "queue": [],
            "collected_total": 0,
            "spent_total": 0,
            "produced_total": 0,
            "dispatched_total": 0,
        }
        cursors[faction] = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "factions": factions,
        "event_cursors": cursors,
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)


def validate_state(state: Any, rules: dict[str, Any]) -> dict[str, Any]:
    if state is None:
        raise PipelineError("E_SEQ_GAP", "initial ledger state is not initialized")
    require(isinstance(state, dict), "E_SEQ_GAP", "initial ledger state must be an object")
    factions = state.get("factions")
    cursors = state.get("event_cursors")
    require(isinstance(factions, dict), "E_SEQ_GAP", "initial ledger has no faction state")
    require(isinstance(cursors, dict), "E_SEQ_GAP", "initial ledger has no event cursors")
    all_factions = set(rules["factions"])
    require(set(factions) == all_factions, "E_SEQ_GAP", "initial ledger faction set is incomplete", expected=sorted(all_factions), actual=sorted(factions))
    require(set(cursors) == all_factions, "E_SEQ_GAP", "initial ledger cursor set is incomplete", expected=sorted(all_factions), actual=sorted(cursors))
    for faction in sorted(all_factions):
        body = factions[faction]
        require(isinstance(body, dict), "E_SEQ_GAP", "faction ledger state must be an object", faction=faction)
        require(isinstance(body.get("resources"), int), "E_SEQ_GAP", "faction resources are missing", faction=faction)
        require(isinstance(body.get("queue"), list), "E_SEQ_GAP", "faction queue is missing", faction=faction)
        require(isinstance(cursors[faction], int), "E_SEQ_GAP", "faction cursor must be an integer", faction=faction)
    return clone_state(state)


def faction_hash(state: dict[str, Any]) -> str:
    return hash_value(
        {
            "resources": state["resources"],
            "queue": state["queue"],
        }
    )

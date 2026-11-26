from __future__ import annotations

import copy
from typing import Any


def find_injection(
    injections: list[dict[str, Any]],
    *,
    stage: str | None = None,
    event_id: str | None = None,
    revision_id: str | None = None,
) -> dict[str, Any] | None:
    for injection in injections:
        if stage is not None and injection.get("stage") != stage:
            continue
        if event_id is not None and injection.get("event_id") != event_id:
            continue
        if revision_id is not None and injection.get("revision_id") != revision_id:
            continue
        return injection
    return None


def apply_pre_validation_injection(
    state: Any, injection: dict[str, Any]
) -> dict[str, Any]:
    """Mutate a cloned workspace before schema validation."""
    event_id = injection.get("event_id")
    fault_type = injection["fault_type"]
    target_arg_id = injection.get("target_arg_id")

    if fault_type == "broken_revision_chain":
        revision_id = injection.get("revision_id")
        for revision in state.revisions_doc["revisions"]:
            if revision["revision_id"] == revision_id:
                revision["parent_revision"] = injection.get(
                    "missing_parent", "rev-injected-missing"
                )
        return injection

    for event in state.events:
        if event.get("event_id") != event_id:
            continue
        if fault_type == "offset_out_of_bounds":
            for argument in event.get("arguments", []):
                if argument.get("arg_id") == target_arg_id:
                    argument["end"] = len(event.get("tokens", [])) + 42
        elif fault_type == "bad_confidence":
            for argument in event.get("arguments", []):
                if argument.get("arg_id") == target_arg_id:
                    argument["confidence_permille"] = 1001
        elif fault_type == "missing_relation_arg":
            relation_id = injection.get("target_relation_id")
            for relation_item in event.get("relations", []):
                if relation_item.get("relation_id") == relation_id:
                    relation_item["target"]["arg_id"] = "INJECTED_MISSING_ARG"
        elif fault_type == "unknown_role":
            for argument in event.get("arguments", []):
                if argument.get("arg_id") == target_arg_id:
                    argument["role"] = injection.get("role", "injected_ghost_role")
        else:
            raise ValueError(f"unknown injection fault type: {fault_type}")
    return injection


def apply_argument_stage_injection(
    event: dict[str, Any], injection: dict[str, Any] | None
) -> dict[str, Any]:
    if injection is None:
        return event
    mutated = copy.deepcopy(event)
    fault_type = injection["fault_type"]
    target_arg_id = injection.get("target_arg_id")
    for argument in mutated.get("arguments", []):
        if argument.get("arg_id") != target_arg_id:
            continue
        if fault_type == "bad_confidence":
            argument["confidence_permille"] = 1001
        elif fault_type == "offset_out_of_bounds":
            argument["end"] = len(mutated.get("tokens", [])) + 42
        elif fault_type == "unknown_role":
            argument["role"] = injection.get("role", "injected_ghost_role")
    return mutated


def apply_reference_stage_injection(
    valid_events: list[dict[str, Any]], injection: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if injection is None:
        return valid_events
    relation_id = injection.get("target_relation_id")
    for event in valid_events:
        if event.get("event_id") != injection.get("event_id"):
            continue
        for relation_item in event.get("relations", []):
            if relation_item.get("relation_id") == relation_id:
                relation_item["target"]["arg_id"] = "INJECTED_MISSING_ARG"
                relation_item["_injection_id"] = injection["injection_id"]
    return valid_events

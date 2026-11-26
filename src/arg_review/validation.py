from __future__ import annotations

from typing import Any

from .canonical import fingerprint


SCHEMA_CODE = "E_SCHEMA_INVALID"


def is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_span(value: Any, token_count: int, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return [f"{label}_not_two_ints"]
    start, end = value
    if not is_plain_int(start) or not is_plain_int(end):
        errors.append(f"{label}_non_integer")
        return errors
    if start < 0 or end < 0 or start > token_count or end > token_count:
        errors.append(f"{label}_out_of_range")
    if start > end:
        errors.append(f"{label}_start_after_end")
    return errors


def validate_revision_chain(doc: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    revisions = doc.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        return {
            "valid": False,
            "errors": ["revision_chain_empty"],
            "revision_map": {},
            "topological_order": [],
            "fingerprint": fingerprint(doc, 16),
        }

    revision_map: dict[str, dict[str, Any]] = {}
    for revision in revisions:
        revision_id = revision.get("revision_id") if isinstance(revision, dict) else None
        if not isinstance(revision_id, str) or not revision_id:
            errors.append("revision_id_missing")
            continue
        if revision_id in revision_map:
            errors.append(f"revision_id_duplicate:{revision_id}")
        revision_map[revision_id] = revision
        if not is_plain_int(revision.get("version")):
            errors.append(f"revision_version_not_int:{revision_id}")

    for revision_id, revision in list(revision_map.items()):
        parent = revision.get("parent_revision")
        if parent is not None and parent not in revision_map:
            errors.append(f"revision_parent_missing:{revision_id}->{parent}")
        if parent is not None and is_plain_int(revision.get("version")):
            parent_revision = revision_map.get(parent)
            if parent_revision and is_plain_int(parent_revision.get("version")):
                if revision["version"] < parent_revision["version"]:
                    errors.append(f"revision_version_decreases:{revision_id}")

    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(revision_id: str) -> None:
        if revision_id in visited or revision_id not in revision_map:
            return
        if revision_id in visiting:
            errors.append(f"revision_cycle_at:{revision_id}")
            return
        visiting.add(revision_id)
        parent = revision_map[revision_id].get("parent_revision")
        if parent:
            visit(parent)
        visiting.remove(revision_id)
        visited.add(revision_id)
        order.append(revision_id)

    for revision_id in sorted(revision_map):
        visit(revision_id)

    active_id = doc.get("active_revision_id")
    if active_id and active_id not in revision_map:
        errors.append(f"active_revision_missing:{active_id}")

    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "revision_map": revision_map,
        "topological_order": order,
        "fingerprint": fingerprint(doc, 16),
    }


def validate_event_shape(
    event: Any,
    vocabulary: set[str],
    revision_map: dict[str, dict[str, Any]],
    active_revision_id: str | None,
    check_roles: bool = True,
) -> list[str]:
    if not isinstance(event, dict):
        return ["entry_not_object"]
    errors: list[str] = []
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        errors.append("event_id_missing")

    tokens = event.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        errors.append("tokens_not_string_list")
        tokens = []
    else:
        unknown = sorted({token for token in tokens if vocabulary and token not in vocabulary})
        if unknown:
            errors.append("token_not_in_controlled_vocabulary:" + ",".join(unknown))

    token_count = len(tokens)
    if "trigger_span" in event:
        errors.extend(_check_span(event.get("trigger_span"), token_count, "trigger_span"))
    elif "trigger_start" in event or "trigger_end" in event:
        errors.extend(
            _check_span(
                [event.get("trigger_start", 0), event.get("trigger_end", 0)],
                token_count,
                "trigger_span",
            )
        )

    schema_event_type = None
    event_type = event.get("event_type")
    active_revision = revision_map.get(active_revision_id or "")
    if active_revision:
        event_types = active_revision.get("event_types", {})
        schema_event_type = event_types.get(event_type) if isinstance(event_types, dict) else None
        if schema_event_type is None:
            errors.append(f"unknown_event_type:{event_type}")

    arguments = event.get("arguments", [])
    if not isinstance(arguments, list):
        errors.append("arguments_not_list")
        arguments = []

    arg_ids: set[str] = set()
    allowed_roles = set(schema_event_type.get("roles", [])) if schema_event_type else set()
    for index, argument in enumerate(arguments):
        prefix = f"argument[{index}]"
        if not isinstance(argument, dict):
            errors.append(f"{prefix}_not_object")
            continue
        arg_id = argument.get("arg_id")
        if not isinstance(arg_id, str) or not arg_id:
            errors.append(f"{prefix}_arg_id_missing")
        elif arg_id in arg_ids:
            errors.append(f"{prefix}_arg_id_duplicate:{arg_id}")
        else:
            arg_ids.add(arg_id)

        role = argument.get("role")
        if not isinstance(role, str) or not role:
            errors.append(f"{prefix}_role_missing")
        elif check_roles and allowed_roles and role not in allowed_roles:
            errors.append(f"{prefix}_role_not_in_set:{role}")

        errors.extend(_check_span([argument.get("start"), argument.get("end")], token_count, f"{prefix}_offset"))

        confidence = argument.get("confidence_permille")
        if not is_plain_int(confidence) or not (0 <= confidence <= 1000):
            errors.append(f"{prefix}_confidence_not_permille")

    relations = event.get("relations", [])
    if not isinstance(relations, list):
        errors.append("relations_not_list")
    else:
        relation_ids: set[str] = set()
        for index, relation_item in enumerate(relations):
            prefix = f"relation[{index}]"
            if not isinstance(relation_item, dict):
                errors.append(f"{prefix}_not_object")
                continue
            relation_id = relation_item.get("relation_id")
            if relation_id in relation_ids:
                errors.append(f"{prefix}_duplicate_id:{relation_id}")
            relation_ids.add(relation_id)
            for endpoint_name in ("source", "target"):
                endpoint = relation_item.get(endpoint_name)
                if not isinstance(endpoint, dict):
                    errors.append(f"{prefix}_{endpoint_name}_missing")
                    continue
                if not endpoint.get("event_id") or not endpoint.get("arg_id"):
                    errors.append(f"{prefix}_{endpoint_name}_incomplete")

    return errors


def validate_workspace(
    events: list[Any],
    vocabulary: set[str],
    revisions_doc: dict[str, Any],
    active_revision_id: str | None = None,
    parse_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chain = validate_revision_chain(revisions_doc)
    revision_map = chain["revision_map"]
    active_id = active_revision_id or revisions_doc.get("active_revision_id")
    invalid_entries: list[dict[str, Any]] = list(parse_errors or [])
    valid_events: list[dict[str, Any]] = []

    raw_event_map: dict[str, dict[str, Any]] = {}
    raw_arg_map: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("event_id"), str):
            raw_event_map.setdefault(event["event_id"], event)
            for argument in event.get("arguments", []):
                if isinstance(argument, dict) and argument.get("arg_id"):
                    raw_arg_map[(event["event_id"], argument["arg_id"])] = argument

    seen_event_ids: set[str] = set()
    for line_index, event in enumerate(events, 1):
        event_id = event.get("event_id") if isinstance(event, dict) else None
        errors = validate_event_shape(
            event, vocabulary, revision_map, active_id, check_roles=chain["valid"]
        )
        if isinstance(event_id, str):
            if event_id in seen_event_ids:
                errors.append(f"event_id_duplicate:{event_id}")
            seen_event_ids.add(event_id)

        if isinstance(event, dict):
            for relation_item in event.get("relations", []) if isinstance(event.get("relations"), list) else []:
                if not isinstance(relation_item, dict):
                    continue
                for endpoint_name in ("source", "target"):
                    endpoint = relation_item.get(endpoint_name, {})
                    if not isinstance(endpoint, dict):
                        continue
                    target_event_id = endpoint.get("event_id")
                    target_arg_id = endpoint.get("arg_id")
                    if target_event_id not in raw_event_map:
                        errors.append(
                            f"relation_{endpoint_name}_event_missing:{target_event_id}"
                        )
                    elif (target_event_id, target_arg_id) not in raw_arg_map:
                        errors.append(
                            f"relation_{endpoint_name}_arg_missing:{target_event_id}/{target_arg_id}"
                        )

        if errors:
            invalid_entries.append(
                {
                    "line_no": line_index,
                    "event_id": event_id,
                    "business_code": SCHEMA_CODE,
                    "errors": sorted(set(errors)),
                }
            )
        else:
            valid_events.append(event)

    valid_events.sort(key=lambda event: event["event_id"])
    invalid_entries.sort(
        key=lambda item: (
            item.get("event_id") or "",
            item.get("line_no") or 0,
            canonical_error_key(item),
        )
    )
    return {
        "valid_events": valid_events,
        "invalid_entries": invalid_entries,
        "revision_chain": chain,
        "raw_event_map": raw_event_map,
        "raw_arg_map": raw_arg_map,
        "event_content_fingerprints": {
            event["event_id"]: fingerprint(
                {"event_id": event["event_id"], "tokens": event["tokens"]}, 16
            )
            for event in valid_events
        },
    }


def canonical_error_key(item: dict[str, Any]) -> str:
    return ",".join(item.get("errors", []))

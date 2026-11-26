from __future__ import annotations

from typing import Any

from .canonical import fingerprint


def _walk_source_revision(
    revision_map: dict[str, dict[str, Any]],
    active_revision: dict[str, Any],
    event_type: str,
    value_fingerprint: str,
    selector: callable,
) -> dict[str, Any]:
    current = active_revision
    while current:
        current_schema = current.get("event_types", {}).get(event_type)
        parent_id = current.get("parent_revision")
        parent = revision_map.get(parent_id) if parent_id else None
        parent_schema = parent.get("event_types", {}).get(event_type) if parent else None
        if parent_schema is not None:
            parent_fp = fingerprint(selector(parent_schema), 16)
            current_fp = fingerprint(selector(current_schema), 16) if current_schema else None
            if current_fp == parent_fp == value_fingerprint:
                current = parent
                continue
        break
    return {
        "rule_revision_id": current["revision_id"],
        "rule_revision_version": current["version"],
        "rule_revision_fingerprint": fingerprint(current, 16),
    }


def build_schema_context(
    revision_map: dict[str, dict[str, Any]], active_revision_id: str
) -> dict[str, dict[str, Any]]:
    active = revision_map[active_revision_id]
    context: dict[str, dict[str, Any]] = {}
    for event_type, schema_type in active.get("event_types", {}).items():
        slice_fp = fingerprint(schema_type, 16)
        trigger_fp = fingerprint(
            {"trigger_terms": sorted(schema_type.get("trigger_terms", []))}, 16
        )
        roles_fp = fingerprint({"roles": sorted(schema_type.get("roles", []))}, 16)
        adjudication_fp = fingerprint(
            {
                "required_roles": sorted(schema_type.get("required_roles", [])),
                "mutex_groups": schema_type.get("mutex_groups", []),
            },
            16,
        )
        slice_source = _walk_source_revision(
            revision_map, active, event_type, slice_fp, lambda value: value
        )
        trigger_source = _walk_source_revision(
            revision_map,
            active,
            event_type,
            trigger_fp,
            lambda value: {"trigger_terms": sorted(value.get("trigger_terms", []))},
        )
        roles_source = _walk_source_revision(
            revision_map,
            active,
            event_type,
            roles_fp,
            lambda value: {"roles": sorted(value.get("roles", []))},
        )
        adjudication_source = _walk_source_revision(
            revision_map,
            active,
            event_type,
            adjudication_fp,
            lambda value: {
                "required_roles": sorted(value.get("required_roles", [])),
                "mutex_groups": value.get("mutex_groups", []),
            },
        )
        context[event_type] = {
            "active_revision_id": active["revision_id"],
            "active_revision_version": active["version"],
            "schema_slice_fingerprint": slice_fp,
            **slice_source,
            "trigger_rule_fingerprint": trigger_fp,
            "trigger_rule_revision_id": trigger_source["rule_revision_id"],
            "trigger_rule_revision_version": trigger_source["rule_revision_version"],
            "role_set_fingerprint": roles_fp,
            "role_rule_revision_id": roles_source["rule_revision_id"],
            "role_rule_revision_version": roles_source["rule_revision_version"],
            "adjudication_rule_fingerprint": adjudication_fp,
            "adjudication_rule_revision_id": adjudication_source["rule_revision_id"],
            "adjudication_rule_revision_version": adjudication_source[
                "rule_revision_version"
            ],
        }
    return context

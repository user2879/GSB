"""Serializable input/parameter anomaly injection."""

from __future__ import annotations

import copy
from typing import Any

from .canonical import content_hash, to_builtin
from .errors import (
    E_INJECT_CONFIG,
    make_error,
)

INJECTION_SCHEMA_VERSION = 1
ALLOWED_KEYS = {
    "schema_version",
    "name",
    "description",
    "degenerate_rays",
    "zero_normals",
    "duplicate_object_ids",
    "priority_overrides",
    "snap_spacing_override",
    "notes",
}


def empty_injection(name: str = "normal", description: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": INJECTION_SCHEMA_VERSION,
        "name": name,
        "description": description or "No anomalies injected.",
        "degenerate_rays": [],
        "zero_normals": [],
        "duplicate_object_ids": [],
        "priority_overrides": [],
        "snap_spacing_override": None,
    }


def validate_injection_spec(
    scene: dict[str, Any],
    query_bundle: dict[str, Any],
    injection: dict[str, Any],
) -> None:
    if not isinstance(injection, dict):
        raise make_error(E_INJECT_CONFIG, "Injection config must be a JSON object.", field="injection")
    unknown = sorted(set(injection) - ALLOWED_KEYS)
    if unknown:
        raise make_error(E_INJECT_CONFIG, f"Unknown injection fields: {unknown}.", fields=unknown)
    if int(injection.get("schema_version", -1)) != INJECTION_SCHEMA_VERSION:
        raise make_error(
            E_INJECT_CONFIG,
            f"Injection schema_version must be {INJECTION_SCHEMA_VERSION}.",
            schema_version=injection.get("schema_version"),
        )

    objects = scene.get("objects", [])
    queries = query_bundle.get("queries", [])
    query_by_id = {str(query["query_id"]): query for query in queries}

    for item in injection.get("degenerate_rays", []):
        if not isinstance(item, dict) or "query_id" not in item:
            raise make_error(E_INJECT_CONFIG, "degenerate_rays entries need query_id.", entry=to_builtin(item))
        query = query_by_id.get(str(item["query_id"]))
        if query is None:
            raise make_error(
                E_INJECT_CONFIG,
                "Degenerate-ray target query does not exist.",
                query_id=str(item["query_id"]),
            )
        if query["kind"] != "ray":
            raise make_error(
                E_INJECT_CONFIG,
                "Degenerate-ray injection can only target ray queries.",
                query_id=str(item["query_id"]),
                query_kind=query["kind"],
            )

    for item in injection.get("zero_normals", []):
        if not isinstance(item, dict) or "object_index" not in item or "plane_index" not in item:
            raise make_error(E_INJECT_CONFIG, "zero_normals entries need object_index and plane_index.", entry=to_builtin(item))
        object_index = int(item["object_index"])
        if object_index < 0 or object_index >= len(objects):
            raise make_error(E_INJECT_CONFIG, "Zero-normal object index is out of range.", object_index=object_index)
        obj = objects[object_index]
        if obj["kind"] != "convex_polyhedron":
            raise make_error(
                E_INJECT_CONFIG,
                "Zero-normal injection targets only convex polyhedra.",
                object_id=obj["id"],
                object_kind=obj["kind"],
            )
        plane_index = int(item["plane_index"])
        planes = obj["geometry"].get("planes", [])
        if plane_index < 0 or plane_index >= len(planes):
            raise make_error(
                E_INJECT_CONFIG,
                "Zero-normal plane index is out of range.",
                object_id=obj["id"],
                plane_index=plane_index,
            )

    for item in injection.get("duplicate_object_ids", []):
        if not isinstance(item, dict) or "object_index" not in item or "new_id" not in item:
            raise make_error(E_INJECT_CONFIG, "duplicate_object_ids entries need object_index and new_id.", entry=to_builtin(item))
        object_index = int(item["object_index"])
        if object_index < 0 or object_index >= len(objects):
            raise make_error(E_INJECT_CONFIG, "Duplicate-ID object index is out of range.", object_index=object_index)
        if not isinstance(item["new_id"], str) or not item["new_id"]:
            raise make_error(E_INJECT_CONFIG, "Duplicate-ID new_id must be a non-empty string.", object_index=object_index)

    for item in injection.get("priority_overrides", []):
        if not isinstance(item, dict) or "object_index" not in item or "priority" not in item:
            raise make_error(E_INJECT_CONFIG, "priority_overrides entries need object_index and priority.", entry=to_builtin(item))
        object_index = int(item["object_index"])
        if object_index < 0 or object_index >= len(objects):
            raise make_error(E_INJECT_CONFIG, "Priority override object index is out of range.", object_index=object_index)
        if isinstance(item["priority"], bool) or not isinstance(item["priority"], int):
            raise make_error(E_INJECT_CONFIG, "Priority override must be an integer.", object_index=object_index)

    override_present = "snap_spacing_override" in injection
    if override_present and injection["snap_spacing_override"] is not None:
        payload = injection["snap_spacing_override"]
        if not isinstance(payload, (int, float, list, tuple)) or isinstance(payload, bool):
            raise make_error(
                E_INJECT_CONFIG,
                "snap_spacing_override must be a number, a three-number list, or null.",
                snap_spacing=to_builtin(payload),
            )
        if isinstance(payload, (list, tuple)) and len(payload) != 3:
            raise make_error(E_INJECT_CONFIG, "snap_spacing_override vector must have three numbers.", snap_spacing=to_builtin(payload))


def apply_injection(
    scene: dict[str, Any],
    query_bundle: dict[str, Any],
    injection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Apply anomalies to deep copies and return a deterministic operation log."""

    validate_injection_spec(scene, query_bundle, injection)
    injected_scene = copy.deepcopy(scene)
    injected_queries = copy.deepcopy(query_bundle)
    query_by_id = {str(query["query_id"]): query for query in injected_queries["queries"]}
    operations: list[dict[str, Any]] = []

    for item in injection.get("degenerate_rays", []):
        query_id = str(item["query_id"])
        query = query_by_id[query_id]
        query["direction"] = [0.0, 0.0, 0.0]
        operations.append(
            {
                "operation": "degenerate_ray",
                "query_id": query_id,
                "before": list(item.get("_before", [])) if isinstance(item.get("_before"), list) else None,
                "after": [0.0, 0.0, 0.0],
            }
        )

    for item in injection.get("zero_normals", []):
        object_index = int(item["object_index"])
        plane_index = int(item["plane_index"])
        obj = injected_scene["objects"][object_index]
        obj["geometry"]["planes"][plane_index]["normal"] = [0.0, 0.0, 0.0]
        operations.append(
            {
                "operation": "zero_normal",
                "object_id": str(obj["id"]),
                "object_index": object_index,
                "plane_index": plane_index,
            }
        )

    for item in injection.get("duplicate_object_ids", []):
        object_index = int(item["object_index"])
        obj = injected_scene["objects"][object_index]
        new_id = str(item["new_id"])
        obj["id"] = new_id
        operations.append(
            {
                "operation": "duplicate_object_id",
                "object_index": object_index,
                "new_id": new_id,
            }
        )

    for item in injection.get("priority_overrides", []):
        object_index = int(item["object_index"])
        priority = int(item["priority"])
        obj = injected_scene["objects"][object_index]
        obj["pick_priority"] = priority
        operations.append(
            {
                "operation": "priority_override",
                "object_id": str(obj["id"]),
                "object_index": object_index,
                "priority": priority,
            }
        )

    if injection.get("snap_spacing_override") is not None:
        spacing = to_builtin(injection["snap_spacing_override"])
        affected: list[str] = []
        for query in injected_queries["queries"]:
            if query["kind"] == "drag":
                query["snap_spacing"] = copy.deepcopy(spacing)
                affected.append(str(query["query_id"]))
        operations.append(
            {
                "operation": "snap_spacing_override",
                "spacing": spacing,
                "query_ids": affected,
            }
        )

    return injected_scene, injected_queries, operations


def injection_hash(injection: dict[str, Any]) -> str:
    return content_hash(injection)

"""Fixed injection sample set used by the repeatable demo."""

from __future__ import annotations

from typing import Any

from .injection import empty_injection


def build_demo_cases(scene: dict[str, Any], query_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    objects = scene["objects"]
    queries = query_bundle["queries"]
    first_ray = next(query for query in queries if query["kind"] == "ray")
    first_poly = next(obj for obj in objects if obj["kind"] == "convex_polyhedron")
    first_drag = next(query for query in queries if query["kind"] == "drag")

    cases: list[dict[str, Any]] = []

    normal = empty_injection("normal", "Baseline scene without injected anomalies.")
    cases.append(
        {
            "name": "00_normal",
            "expected_error_codes": [],
            "injection": normal,
        }
    )

    degenerate = empty_injection(
        "degenerate_ray",
        "Replace one ray direction with the zero vector.",
    )
    degenerate["degenerate_rays"] = [{"query_id": first_ray["query_id"]}]
    cases.append(
        {
            "name": "01_degenerate_ray",
            "expected_error_codes": ["E_RAY_DEGENERATE"],
            "injection": degenerate,
        }
    )

    zero_normal = empty_injection(
        "zero_normal",
        "Replace one convex-polyhedron plane normal with the zero vector.",
    )
    zero_normal["zero_normals"] = [
        {"object_index": int(first_poly["object_index"]), "plane_index": 0}
    ]
    cases.append(
        {
            "name": "02_zero_normal",
            "expected_error_codes": ["E_NORMAL_ZERO"],
            "injection": zero_normal,
        }
    )

    duplicate = empty_injection(
        "duplicate_object_id",
        "Give one object the same ID as object zero.",
    )
    duplicate["duplicate_object_ids"] = [{"object_index": 1, "new_id": objects[0]["id"]}]
    cases.append(
        {
            "name": "03_duplicate_id",
            "expected_error_codes": ["E_OBJECT_DUP_ID"],
            "injection": duplicate,
        }
    )

    priority = empty_injection(
        "priority_range",
        "Set one object's pick priority above the declared maximum.",
    )
    priority["priority_overrides"] = [{"object_index": 0, "priority": 999}]
    cases.append(
        {
            "name": "04_priority_range",
            "expected_error_codes": ["E_PRIORITY_RANGE"],
            "injection": priority,
        }
    )

    spacing = empty_injection(
        "snap_spacing",
        "Supply unequal per-axis snap spacing to every drag query.",
    )
    spacing["snap_spacing_override"] = [0.25, 0.20, 0.25]
    cases.append(
        {
            "name": "05_snap_spacing",
            "expected_error_codes": ["E_SNAP_SPACING"],
            "injection": spacing,
        }
    )

    invalid = empty_injection(
        "invalid_inject_config",
        "Reference a query that does not exist.",
    )
    invalid["degenerate_rays"] = [{"query_id": "q-does-not-exist"}]
    cases.append(
        {
            "name": "06_invalid_inject_config",
            "expected_error_codes": ["E_INJECT_CONFIG"],
            "injection": invalid,
        }
    )

    return cases

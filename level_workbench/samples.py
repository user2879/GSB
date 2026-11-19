"""Locally synthesized specifications and injection configurations."""

from __future__ import annotations

import copy
from typing import Any


def normal_spec() -> dict[str, Any]:
    return {
        "schema": "level-workbench/spec-v1",
        "seed": 20261119,
        "grid": {"width": 20, "height": 14},
        "terrain": {
            "weights": {
                "grass": 0.58,
                "sand": 0.20,
                "rock": 0.14,
                "water": 0.08,
            },
            "blocked": ["water"],
        },
        "regions": [
            {"id": "grove", "bounds": [2, 2, 6, 5]},
            {"id": "ridge", "bounds": [12, 2, 17, 5]},
            {"id": "marsh", "bounds": [13, 9, 18, 12]},
        ],
        "resources": {"count": [10, 14], "allowed_terrain": ["grass", "sand"]},
        "spawn": [1, 7],
        "target": [18, 7],
        "forbidden_adjacent": [["rock", "water"]],
        "fixed_cells": [],
    }


def connectivity_unsat_spec() -> dict[str, Any]:
    spec = normal_spec()
    spec["seed"] = 20261120
    spec["grid"] = {"width": 16, "height": 10}
    spec["regions"] = [
        {"id": "far_grove", "bounds": [11, 2, 14, 6]},
        {"id": "near_field", "bounds": [3, 1, 5, 3]},
    ]
    spec["resources"] = {"count": [6, 8], "allowed_terrain": ["grass", "sand"]}
    spec["spawn"] = [2, 5]
    spec["target"] = [13, 5]
    spec["fixed_cells"] = [
        {"coord": [8, y], "terrain": "water"}
        for y in range(10)
    ]
    return spec


def injection_configs() -> dict[str, dict[str, Any]]:
    def controlled_forbidden_pair() -> list[dict[str, Any]]:
        entries = [
            {"coord": [7, 6], "terrain": "grass"},
            {"coord": [8, 5], "terrain": "grass"},
            {"coord": [9, 5], "terrain": "grass"},
            {"coord": [10, 6], "terrain": "grass"},
            {"coord": [8, 7], "terrain": "grass"},
            {"coord": [9, 7], "terrain": "grass"},
            {"coord": [8, 6], "terrain": "rock"},
            {"coord": [9, 6], "terrain": "water"},
        ]
        return sorted(entries, key=lambda entry: (entry["coord"][1], entry["coord"][0]))

    spec_overflow = {
        "schema": "level-workbench/injection-v1",
        "id": "spec-resource-density-overflow",
        "description": "Tampers resources.count beyond the grid cell count.",
        "spec_patches": [
            {"op": "set", "path": "resources.count", "value": [300, 320]},
        ],
        "pipeline": {},
        "artifact_tamper": {},
    }
    pipeline_overflow = {
        "schema": "level-workbench/injection-v1",
        "id": "pipeline-resource-count-overflow",
        "description": "Forces 20 resources although the spec permits at most 14.",
        "spec_patches": [],
        "pipeline": {"resource_count_override": 20, "skip_resource_repair": True},
        "artifact_tamper": {},
    }
    illegal_adjacent = {
        "schema": "level-workbench/injection-v1",
        "id": "pipeline-forced-forbidden-adjacency",
        "description": "After normal repair, forces rock next to water at [8,6]/[9,6].",
        "spec_patches": [],
        "pipeline": {
            "post_generation_force_terrain": controlled_forbidden_pair()
        },
        "artifact_tamper": {},
    }
    subseed_offset = {
        "schema": "level-workbench/injection-v1",
        "id": "pipeline-terrain-subseed-offset",
        "description": "Offsets only the terrain sub-seed; used to prove purpose isolation.",
        "spec_patches": [],
        "pipeline": {"subseed_offsets": {"terrain": 17}},
        "artifact_tamper": {},
    }
    verifier_mismatch = {
        "schema": "level-workbench/injection-v1",
        "id": "artifact-tamper-after-generator-evidence",
        "description": "Mutates the serialized artifact after generation so evidence and verification diverge.",
        "spec_patches": [],
        "pipeline": {},
        "artifact_tamper": {
            "force_terrain": controlled_forbidden_pair()
        },
    }
    return {
        "resource_spec_overflow": spec_overflow,
        "resource_pipeline_overflow": pipeline_overflow,
        "illegal_adjacent": illegal_adjacent,
        "terrain_subseed_offset": subseed_offset,
        "verifier_mismatch": verifier_mismatch,
    }


def demo_cases() -> list[dict[str, Any]]:
    injections = injection_configs()
    return [
        {
            "id": "normal",
            "kind": "normal",
            "spec": normal_spec(),
            "injection": None,
        },
        {
            "id": "connectivity_unsat",
            "kind": "connectivity_unsat",
            "spec": connectivity_unsat_spec(),
            "injection": None,
        },
        {
            "id": "resource_spec_overflow",
            "kind": "spec_injection",
            "spec": normal_spec(),
            "injection": injections["resource_spec_overflow"],
        },
        {
            "id": "resource_pipeline_overflow",
            "kind": "pipeline_injection",
            "spec": normal_spec(),
            "injection": injections["resource_pipeline_overflow"],
        },
        {
            "id": "illegal_adjacent",
            "kind": "pipeline_injection",
            "spec": normal_spec(),
            "injection": injections["illegal_adjacent"],
        },
        {
            "id": "verifier_mismatch",
            "kind": "artifact_tamper",
            "spec": normal_spec(),
            "injection": injections["verifier_mismatch"],
        },
    ]


def clone_case(case: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(case)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def deterministic_mesh_payload() -> dict[str, Any]:
    """A small defective mesh: hole, non-manifold edge, flip, degenerate face, isolated point."""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.35, -0.15, 0.55],
        [2.25, 2.25, 2.25],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 5, 6],  # deliberately reversed right-side triangle
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
        [0, 1, 8],  # third face incident on edge 0-1
        [0, 0, 1],  # repeated-index degenerate face
    ]
    return {"vertices": vertices, "faces": faces}


def deterministic_repair_script() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "coordinate_tolerance": 1.0e-7,
        "area_tolerance": 1.0e-10,
        "fixed_scale": 1_000_000,
        "trace_radius": 5,
        "concurrency": 4,
        "memory_budget_nodes": 8192,
        "operators": [
            {"type": "hole_triangulation"},
            {"type": "remove_degenerate_faces"},
            {"type": "reorient_normals"},
            {"type": "laplacian_smooth", "iterations": 16, "lambda": 0.42},
        ],
    }


def write_sample(directory: str | Path) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    mesh_path = target / "sample_defective.mesh.json"
    script_path = target / "sample_repair_script.json"
    mesh_path.write_text(
        json.dumps(deterministic_mesh_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    script_path.write_text(
        json.dumps(deterministic_repair_script(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return mesh_path, script_path

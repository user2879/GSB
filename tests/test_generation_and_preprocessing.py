from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mesh_repair.config import load_config
from mesh_repair.generator import baseline_stats, generate_sample
from mesh_repair.hashing import canonical_json_bytes
from mesh_repair.mesh import (
    edge_diagnostics,
    find_self_intersections,
    mesh_components,
    triangle_areas,
)
from mesh_repair.preprocessing import (
    partition_blocks,
    validate_chunk_manifest,
    weld_vertices,
)

from .conftest import PROJECT_ROOT


def test_generator_is_byte_deterministic_and_defects_are_real(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "default.json")
    first = generate_sample(config, tmp_path / "first")
    second = generate_sample(config, tmp_path / "second")

    for filename in ("baseline.ply", "defective.ply", "ground_truth.json"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()

    truth = first.truth
    assert truth["injection_count"] == config.defects.injection_count
    assert truth["defect_record_count"] == config.defects.injection_count
    assert len(truth["defects"]) == config.defects.injection_count
    assert sum(truth["counts_by_type"].values()) == config.defects.injection_count
    assert Counter(record["type"] for record in truth["defects"]) == Counter(truth["counts_by_type"])

    baseline_components = baseline_stats(first.baseline).component_count
    defective_components = mesh_components(first.defective.faces)[1]
    assert baseline_components == 3
    assert defective_components > baseline_components
    assert edge_diagnostics(first.defective.faces)["nonmanifold_edge_count"] > 0
    assert np.any(triangle_areas(first.defective.vertices, first.defective.faces) <= 1.0e-14)
    assert find_self_intersections(
        first.defective.vertices,
        first.defective.faces,
        config.self_intersection_epsilon,
    )


def test_weld_and_chunk_manifest_are_deterministic(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "default.json")
    generation = generate_sample(config, tmp_path / "sample")
    weld = weld_vertices(
        generation.defective.vertices,
        generation.defective.faces,
        config.weld_tolerance,
    )
    assert len(weld.removed_vertices) == config.defects.duplicate_vertex

    manifest_a = partition_blocks(
        weld.mesh,
        weld_tolerance=config.weld_tolerance,
        max_faces_per_block=config.max_faces_per_block,
    )
    manifest_b = partition_blocks(
        weld.mesh,
        weld_tolerance=config.weld_tolerance,
        max_faces_per_block=config.max_faces_per_block,
    )
    assert canonical_json_bytes(manifest_a.to_dict()) == canonical_json_bytes(manifest_b.to_dict())
    validate_chunk_manifest(manifest_a.to_dict(), weld.mesh.vertex_count, weld.mesh.face_count)

    covered_faces = [face for block in manifest_a.blocks for face in block.face_indices]
    assert sorted(covered_faces) == list(range(weld.mesh.face_count))
    assert len(manifest_a.shared_boundary_vertices) > 0
    boundary_users: Counter[int] = Counter()
    for block in manifest_a.blocks:
        assert len(block.face_indices) <= config.max_faces_per_block
        boundary_users.update(block.boundary_vertices)
    assert all(boundary_users[vertex] >= 2 for vertex in manifest_a.shared_boundary_vertices)


def test_duplicate_block_key_is_rejected(tmp_path):
    config = load_config(PROJECT_ROOT / "configs" / "default.json")
    generation = generate_sample(config, tmp_path / "sample")
    weld = weld_vertices(
        generation.defective.vertices,
        generation.defective.faces,
        config.weld_tolerance,
    )
    manifest = partition_blocks(
        weld.mesh,
        weld_tolerance=config.weld_tolerance,
        max_faces_per_block=config.max_faces_per_block,
    ).to_dict()
    manifest["blocks"][1]["key"] = manifest["blocks"][0]["key"]
    with pytest.raises(Exception) as exc_info:
        validate_chunk_manifest(manifest, weld.mesh.vertex_count, weld.mesh.face_count)
    assert getattr(exc_info.value, "code", None).value == "E_BLOCK_KEY"

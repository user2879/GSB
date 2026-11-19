"""Deterministic vertex welding and repair-block partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BusinessError, ErrorCode
from .hashing import write_json
from .mesh import Mesh, mesh_components


@dataclass(frozen=True)
class DuplicateGroup:
    canonical_vertex: int
    duplicate_vertices: tuple[int, ...]


@dataclass(frozen=True)
class WeldResult:
    mesh: Mesh
    original_to_welded: np.ndarray
    welded_to_original: np.ndarray
    removed_vertices: tuple[int, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_vertex_count": int(len(self.original_to_welded)),
            "output_vertex_count": self.mesh.vertex_count,
            "removed_vertex_count": len(self.removed_vertices),
            "removed_vertices": list(self.removed_vertices),
            "duplicate_groups": [
                {
                    "canonical_vertex": group.canonical_vertex,
                    "duplicate_vertices": list(group.duplicate_vertices),
                }
                for group in self.duplicate_groups
            ],
        }


def weld_vertices(vertices: np.ndarray, faces: np.ndarray, tolerance: float) -> WeldResult:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    count = len(vertices)
    parent = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    if tolerance <= 0:
        exact: dict[tuple[float, float, float], int] = {}
        for index, point in enumerate(vertices):
            key = (float(point[0]), float(point[1]), float(point[2]))
            if key in exact:
                union(exact[key], index)
            else:
                exact[key] = index
    else:
        tolerance_squared = tolerance * tolerance
        for index in range(count):
            if index == 0:
                continue
            differences = vertices[:index] - vertices[index]
            distances_squared = np.einsum("ij,ij->i", differences, differences)
            for candidate in np.nonzero(distances_squared <= tolerance_squared)[0]:
                union(int(candidate), index)

    roots = np.array([find(index) for index in range(count)], dtype=np.int64)
    unique_roots = np.unique(roots)
    root_to_new = {int(root): new for new, root in enumerate(unique_roots)}
    original_to_welded = np.array(
        [root_to_new[int(root)] for root in roots], dtype=np.int64
    )
    welded_vertices = vertices[unique_roots]
    welded_faces = original_to_welded[faces]

    members: dict[int, list[int]] = {int(root): [] for root in unique_roots}
    for index, root in enumerate(roots):
        members[int(root)].append(index)
    duplicate_groups = tuple(
        DuplicateGroup(
            canonical_vertex=int(root),
            duplicate_vertices=tuple(value for value in members[int(root)] if value != int(root)),
        )
        for root in unique_roots
        if len(members[int(root)]) > 1
    )
    removed = tuple(
        index for index, root in enumerate(roots) if int(root) != index
    )
    return WeldResult(
        mesh=Mesh(vertices=welded_vertices, faces=welded_faces),
        original_to_welded=original_to_welded,
        welded_to_original=unique_roots.astype(np.int64, copy=False),
        removed_vertices=removed,
        duplicate_groups=duplicate_groups,
    )


@dataclass(frozen=True)
class RepairBlock:
    key: str
    component: int
    part: int
    face_indices: tuple[int, ...]
    vertex_indices: tuple[int, ...]
    boundary_vertices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "component": self.component,
            "part": self.part,
            "face_count": len(self.face_indices),
            "face_indices": list(self.face_indices),
            "vertex_indices": list(self.vertex_indices),
            "boundary_vertices": list(self.boundary_vertices),
        }


@dataclass(frozen=True)
class ChunkManifest:
    weld_tolerance: float
    max_faces_per_block: int
    component_count: int
    split_rule: str
    blocks: tuple[RepairBlock, ...]
    shared_boundary_vertices: tuple[int, ...]
    vertex_count: int
    face_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "split_rule": self.split_rule,
            "weld_tolerance": self.weld_tolerance,
            "max_faces_per_block": self.max_faces_per_block,
            "component_count": self.component_count,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "block_count": len(self.blocks),
            "shared_boundary_vertex_count": len(self.shared_boundary_vertices),
            "shared_boundary_vertices": list(self.shared_boundary_vertices),
            "blocks": [block.to_dict() for block in self.blocks],
        }


def partition_blocks(
    mesh: Mesh,
    *,
    weld_tolerance: float,
    max_faces_per_block: int,
) -> ChunkManifest:
    labels, component_count = mesh_components(mesh.faces)
    component_faces = [
        sorted(int(value) for value in np.nonzero(labels == component)[0])
        for component in range(component_count)
    ]
    component_faces.sort(key=lambda values: values[0] if values else 10**18)

    raw_blocks: list[tuple[int, int, list[int]]] = []
    for component_index, face_ids in enumerate(component_faces):
        for part, start in enumerate(range(0, len(face_ids), max_faces_per_block)):
            raw_blocks.append(
                (component_index, part, face_ids[start : start + max_faces_per_block])
            )
    raw_blocks.sort(key=lambda value: (value[0], value[1]))

    block_vertices: list[list[int]] = []
    for component_index, _part, face_ids in raw_blocks:
        used = sorted({int(vertex) for face in face_ids for vertex in mesh.faces[face]})
        block_vertices.append(used)
    # Boundary membership is computed per component, using a synthetic global
    # key (component, vertex) because welded components do not share vertices.
    component_vertex_parts: dict[tuple[int, int], set[int]] = {}
    for block_index, (component_index, _part, _face_ids) in enumerate(raw_blocks):
        for vertex in block_vertices[block_index]:
            component_vertex_parts.setdefault((component_index, vertex), set()).add(block_index)

    blocks: list[RepairBlock] = []
    global_boundary: set[int] = set()
    for block_index, (component_index, part, face_ids) in enumerate(raw_blocks):
        boundary = tuple(
            vertex
            for vertex in block_vertices[block_index]
            if len(component_vertex_parts[(component_index, vertex)]) > 1
        )
        global_boundary.update(boundary)
        blocks.append(
            RepairBlock(
                key=f"C{component_index:05d}-P{part:05d}",
                component=component_index,
                part=part,
                face_indices=tuple(face_ids),
                vertex_indices=tuple(block_vertices[block_index]),
                boundary_vertices=boundary,
            )
        )

    blocks.sort(key=lambda block: block.key)
    manifest = ChunkManifest(
        weld_tolerance=weld_tolerance,
        max_faces_per_block=max_faces_per_block,
        component_count=component_count,
        split_rule="connected_component_then_sorted_face_id_slices",
        blocks=tuple(blocks),
        shared_boundary_vertices=tuple(sorted(global_boundary)),
        vertex_count=mesh.vertex_count,
        face_count=mesh.face_count,
    )
    validate_chunk_manifest(manifest.to_dict(), mesh.vertex_count, mesh.face_count)
    return manifest


def validate_chunk_manifest(
    manifest: dict[str, Any], vertex_count: int, face_count: int
) -> None:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise BusinessError(ErrorCode.E_BLOCK_KEY, "chunk manifest has no blocks")
    seen_keys: set[str] = set()
    seen_faces: set[int] = set()
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise BusinessError(
                ErrorCode.E_BLOCK_KEY,
                "block entry must be an object",
                location={"block_index": block_index},
            )
        key = block.get("key")
        if not isinstance(key, str) or not key:
            raise BusinessError(
                ErrorCode.E_BLOCK_KEY,
                "block key is missing or empty",
                location={"block_index": block_index},
            )
        if key in seen_keys:
            raise BusinessError(
                ErrorCode.E_BLOCK_KEY,
                "duplicate block key",
                location={"block": key},
            )
        seen_keys.add(key)
        faces = block.get("face_indices")
        vertices = block.get("vertex_indices")
        boundary = block.get("boundary_vertices")
        if not isinstance(faces, list) or not faces:
            raise BusinessError(ErrorCode.E_BLOCK_KEY, "block has no faces", location={"block": key})
        if not isinstance(vertices, list) or not isinstance(boundary, list):
            raise BusinessError(ErrorCode.E_BLOCK_KEY, "invalid block vertex lists", location={"block": key})
        for face in faces:
            if not isinstance(face, int) or face < 0 or face >= face_count:
                raise BusinessError(
                    ErrorCode.E_BLOCK_KEY,
                    "block face index is outside the mesh",
                    location={"block": key, "face": face},
                )
            if face in seen_faces:
                raise BusinessError(
                    ErrorCode.E_BLOCK_KEY,
                    "face belongs to more than one block",
                    location={"block": key, "face": face},
                )
            seen_faces.add(face)
        vertex_set = set(vertices)
        for vertex in vertices:
            if not isinstance(vertex, int) or vertex < 0 or vertex >= vertex_count:
                raise BusinessError(
                    ErrorCode.E_BLOCK_KEY,
                    "block vertex index is outside the mesh",
                    location={"block": key, "vertex": vertex},
                )
        for vertex in boundary:
            if vertex not in vertex_set:
                raise BusinessError(
                    ErrorCode.E_BLOCK_KEY,
                    "boundary vertex is not part of its block",
                    location={"block": key, "vertex": vertex},
                )
    missing = sorted(set(range(face_count)).difference(seen_faces))
    if missing:
        raise BusinessError(
            ErrorCode.E_BLOCK_KEY,
            "chunk manifest does not cover every face",
            location={"first_missing_face": missing[0]},
        )


def write_chunk_manifest(path: str | Path, manifest: ChunkManifest) -> bytes:
    return write_json(path, manifest.to_dict())

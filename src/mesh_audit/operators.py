from __future__ import annotations

from collections import deque
from fractions import Fraction
from typing import Any

import numpy as np

from .fixedpoint import (
    laplacian_fixed_row,
    laplacian_float_row,
    mean_fixed_row,
    mean_float_row,
)
from .mesh import MeshState
from .topology import TopologyData, face_edges, sorted_edge


def _boundary_components(topology: TopologyData) -> list[tuple[int, ...]]:
    neighbors: dict[int, list[int]] = {}
    for a, b in topology.boundary_edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    for values in neighbors.values():
        values.sort()
    remaining = set(neighbors)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        component: list[int] = []
        while queue:
            vertex = queue.popleft()
            component.append(vertex)
            for neighbor in neighbors[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda values: values[0])
    return components


def _extract_cycle(
    component: tuple[int, ...], topology: TopologyData
) -> tuple[int, ...] | None:
    boundary_set = set(topology.boundary_edges)
    neighbors: dict[int, list[int]] = {vertex: [] for vertex in component}
    for a, b in topology.boundary_edges:
        if a in neighbors and b in neighbors:
            neighbors[a].append(b)
            neighbors[b].append(a)
    for values in neighbors.values():
        values.sort()
    if any(len(values) != 2 for values in neighbors.values()):
        return None
    start = component[0]
    loop = [start]
    previous = -1
    current = start
    used: set[tuple[int, int]] = set()
    while True:
        candidates = [value for value in neighbors[current] if value != previous]
        if not candidates:
            return None
        nxt = candidates[0]
        edge = sorted_edge(current, nxt)
        if edge in used:
            return None
        used.add(edge)
        if nxt == start:
            break
        loop.append(nxt)
        previous, current = current, nxt
        if len(loop) > len(component):
            return None
    if len(used) != len(component) or any(edge not in boundary_set for edge in used):
        return None
    return tuple(loop)


def _directed_boundary_map(topology: TopologyData, faces: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    boundary = set(topology.boundary_edges)
    result: dict[tuple[int, int], tuple[int, int]] = {}
    for face_index, face in enumerate(faces.tolist()):
        if bool(topology.degenerate_mask[face_index]):
            continue
        a, b, c = (int(value) for value in face)
        for left, right in ((a, b), (b, c), (c, a)):
            edge = sorted_edge(left, right)
            if edge in boundary and edge not in result:
                result[edge] = (left, right)
    return result


def _orient_hole_loop(
    loop: tuple[int, ...], directed_boundary: dict[tuple[int, int], tuple[int, int]]
) -> tuple[int, ...]:
    first = loop[0]
    second = loop[1] if len(loop) > 1 else loop[0]
    desired_existing = directed_boundary.get(sorted_edge(first, second))
    if desired_existing == (first, second):
        return tuple(reversed(loop))
    return loop


def hole_triangulation(state: MeshState, topology: TopologyData) -> tuple[MeshState, list[int], dict[str, Any]]:
    fixed_rows = [tuple(int(value) for value in row) for row in state.fixed.tolist()]
    reference_rows = [tuple(float(value) for value in row) for row in state.reference.tolist()]
    faces = [tuple(int(value) for value in face) for face in state.faces.tolist()]
    directed = _directed_boundary_map(topology, state.faces)
    loops_filled = 0
    loops_skipped = 0
    faces_added = 0
    vertices_added = 0
    source_vertices: set[int] = set()

    for component in _boundary_components(topology):
        loop = _extract_cycle(component, topology)
        if loop is None or len(loop) < 3:
            loops_skipped += 1
            continue
        loop = _orient_hole_loop(loop, directed)
        loop_fixed = np.asarray([fixed_rows[vertex] for vertex in loop], dtype=np.int64)
        loop_reference = np.asarray([reference_rows[vertex] for vertex in loop], dtype=np.float64)
        centroid_fixed = mean_fixed_row(loop_fixed)
        centroid_reference = mean_float_row(loop_reference)
        new_vertex = len(fixed_rows)
        fixed_rows.append(centroid_fixed)
        reference_rows.append(centroid_reference)
        vertices_added += 1
        source_vertices.add(new_vertex)
        source_vertices.update(int(vertex) for vertex in loop)
        for index, left in enumerate(loop):
            right = loop[(index + 1) % len(loop)]
            faces.append((left, right, new_vertex))
            faces_added += 1
        loops_filled += 1

    result = MeshState(
        fixed=np.asarray(fixed_rows, dtype=np.int64).reshape((-1, 3)),
        reference=np.asarray(reference_rows, dtype=np.float64).reshape((-1, 3)),
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        scale=state.scale,
        source_vertices=tuple(sorted(source_vertices)),
    )
    details = {
        "loops_filled": loops_filled,
        "loops_skipped": loops_skipped,
        "faces_added": faces_added,
        "vertices_added": vertices_added,
    }
    return result, sorted(source_vertices), details


def remove_degenerate_faces(
    state: MeshState, topology: TopologyData
) -> tuple[MeshState, list[int], dict[str, Any]]:
    mask = topology.degenerate_mask
    removed_indices = [int(index) for index in np.flatnonzero(mask)]
    sources: set[int] = set()
    for index in removed_indices:
        sources.update(int(vertex) for vertex in state.faces[index].tolist())
    kept = ~mask
    result = MeshState(
        fixed=state.fixed.copy(),
        reference=state.reference.copy(),
        faces=state.faces[kept].copy(),
        scale=state.scale,
        source_vertices=tuple(sorted(sources)),
    )
    details = {
        "removed_face_count": len(removed_indices),
        "removed_face_indices": removed_indices,
        "remaining_face_count": int(result.faces.shape[0]),
    }
    return result, sorted(sources), details


def _face_direction(face: tuple[int, int, int], edge: tuple[int, int]) -> int | None:
    a, b, c = face
    u, v = edge
    if (a == u and b == v) or (b == u and c == v) or (c == u and a == v):
        return 1
    if (a == v and b == u) or (b == v and c == u) or (c == v and a == u):
        return -1
    return None


def _component_six_volume(fixed: np.ndarray, faces: list[tuple[int, int, int]]) -> int:
    points = fixed.tolist()
    total = 0
    for face in faces:
        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]
        total += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return int(total)


def reorient_normals(
    state: MeshState, topology: TopologyData
) -> tuple[MeshState, list[int], dict[str, Any]]:
    faces = [tuple(int(value) for value in face) for face in state.faces.tolist()]
    adjacency = topology.face_adjacency
    visited = [False] * len(faces)
    flip = [False] * len(faces)
    components: list[list[int]] = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                shared = set(face_edges(faces[current])).intersection(
                    face_edges(faces[neighbor])
                )
                if len(shared) != 1:
                    continue
                edge = next(iter(shared))
                parent_direction = _face_direction(faces[current], edge)
                child_direction = _face_direction(faces[neighbor], edge)
                if parent_direction is None or child_direction is None:
                    continue
                parent_effective = parent_direction * (-1 if flip[current] else 1)
                parent_original = parent_direction
                flip[neighbor] = flip[current] ^ (parent_original == child_direction)
                if parent_effective == 0:
                    raise RuntimeError("unreachable orientation condition")
                visited[neighbor] = True
                queue.append(neighbor)
        component.sort()
        components.append(component)

    for component in components:
        volume = 0
        for face_index in component:
            face = faces[face_index]
            if flip[face_index]:
                face = (face[0], face[2], face[1])
            volume += _component_six_volume(
                state.fixed, [face]
            )
        if volume < 0:
            for face_index in component:
                flip[face_index] = not flip[face_index]

    flipped_sources: set[int] = set()
    flipped_indices: list[int] = []
    for face_index, should_flip in enumerate(flip):
        if not should_flip:
            continue
        a, b, c = faces[face_index]
        faces[face_index] = (a, c, b)
        flipped_indices.append(face_index)
        flipped_sources.update((a, b, c))

    result = MeshState(
        fixed=state.fixed.copy(),
        reference=state.reference.copy(),
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        scale=state.scale,
        source_vertices=tuple(sorted(flipped_sources)),
    )
    details = {
        "reoriented_face_count": len(flipped_indices),
        "reoriented_face_indices": flipped_indices,
        "face_component_count": len(components),
    }
    return result, sorted(flipped_sources), details


def laplacian_smooth(
    state: MeshState,
    topology: TopologyData,
    iterations: int,
    lam: float,
) -> tuple[MeshState, list[int], dict[str, Any]]:
    fixed = state.fixed.copy()
    reference = state.reference.copy()
    neighbors = topology.vertex_neighbors
    lambda_fraction = Fraction(str(lam))
    changed_union: set[int] = set()
    per_iteration_changed: list[int] = []
    total_axis_updates = 0

    for _ in range(iterations):
        old_fixed = fixed.copy()
        old_reference = reference.copy()
        new_fixed = fixed.copy()
        new_reference = reference.copy()
        changed: set[int] = set()
        for vertex in range(int(fixed.shape[0])):
            neighbor_ids = neighbors[vertex]
            degree = len(neighbor_ids)
            if degree == 0:
                continue
            fixed_sum = tuple(
                sum(int(old_fixed[neighbor, axis]) for neighbor in neighbor_ids)
                for axis in range(3)
            )
            reference_sum = tuple(
                float(np.sum(old_reference[np.asarray(neighbor_ids, dtype=np.int64), axis], dtype=np.float64))
                for axis in range(3)
            )
            fixed_result = laplacian_fixed_row(
                tuple(int(value) for value in old_fixed[vertex]),
                fixed_sum,
                degree,
                lambda_fraction,
            )
            float_result = laplacian_float_row(
                tuple(float(value) for value in old_reference[vertex]),
                reference_sum,
                degree,
                lam,
            )
            if fixed_result != tuple(int(value) for value in old_fixed[vertex]):
                changed.add(vertex)
            if float_result != tuple(float(value) for value in old_reference[vertex]):
                changed.add(vertex)
            new_fixed[vertex] = fixed_result
            new_reference[vertex] = float_result
        changed_union.update(changed)
        per_iteration_changed.append(len(changed))
        total_axis_updates += len(changed) * 3
        fixed, reference = new_fixed, new_reference

    result = MeshState(
        fixed=fixed,
        reference=reference,
        faces=state.faces.copy(),
        scale=state.scale,
        source_vertices=tuple(sorted(changed_union)),
    )
    details = {
        "iterations": iterations,
        "lambda": lam,
        "changed_vertices_per_iteration": per_iteration_changed,
        "coordinate_updates": total_axis_updates,
        "fixed_lambda_numerator": lambda_fraction.numerator,
        "fixed_lambda_denominator": lambda_fraction.denominator,
    }
    return result, sorted(changed_union), details


def apply_operator(
    state: MeshState,
    topology: TopologyData,
    operator: dict[str, Any],
) -> tuple[MeshState, list[int], dict[str, Any], int]:
    operator_type = str(operator["type"])
    if operator_type == "hole_triangulation":
        result, sources, details = hole_triangulation(state, topology)
        execution_count = int(details["loops_filled"])
    elif operator_type == "remove_degenerate_faces":
        result, sources, details = remove_degenerate_faces(state, topology)
        execution_count = int(details["removed_face_count"])
    elif operator_type == "reorient_normals":
        result, sources, details = reorient_normals(state, topology)
        execution_count = int(details["reoriented_face_count"])
    elif operator_type == "laplacian_smooth":
        result, sources, details = laplacian_smooth(
            state,
            topology,
            int(operator["iterations"]),
            float(operator["lambda"]),
        )
        execution_count = int(operator["iterations"])
    else:
        raise ValueError(f"unsupported operator: {operator_type}")
    if not np.all(np.isfinite(result.reference)):
        raise FloatingPointError("float64 reference coordinates became non-finite")
    return result, sources, details, execution_count

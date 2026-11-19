from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .fixedpoint import fixed_as_float


Edge = tuple[int, int]


def sorted_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def face_edges(face: Iterable[int]) -> tuple[Edge, Edge, Edge]:
    a, b, c = (int(value) for value in face)
    return (sorted_edge(a, b), sorted_edge(b, c), sorted_edge(c, a))


def unique_face_edges(face: Iterable[int]) -> tuple[Edge, ...]:
    return tuple(sorted(set(face_edges(face))))


def degenerate_face_mask(
    fixed: np.ndarray, faces: np.ndarray, scale: int, area_tolerance: float
) -> np.ndarray:
    """A face is degenerate for repeated indices or an area at/below tolerance."""

    count = int(faces.shape[0])
    result = np.zeros(count, dtype=bool)
    if count == 0:
        return result
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    result |= repeated
    coordinates = fixed_as_float(fixed, scale)
    points = coordinates[faces]
    edge_a = points[:, 1] - points[:, 0]
    edge_b = points[:, 2] - points[:, 0]
    cross = np.cross(edge_a, edge_b)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    result |= area <= area_tolerance
    return result


@dataclass(slots=True)
class TopologyData:
    vertex_count: int
    face_count: int
    edge_count: int
    euler_characteristic: int
    degenerate_mask: np.ndarray
    vertex_faces: tuple[tuple[int, ...], ...]
    face_adjacency: tuple[tuple[int, ...], ...]
    vertex_neighbors: tuple[tuple[int, ...], ...]
    edge_face_map: dict[Edge, tuple[int, ...]]
    boundary_edges: tuple[Edge, ...]
    boundary_loop_count: int
    boundary_component_count: int
    open_boundary_chain_count: int
    nonmanifold_edges: tuple[Edge, ...]
    orientation_conflict_edges: tuple[Edge, ...]
    isolated_vertices: tuple[int, ...]
    rescan_matches: bool

    def summary(self) -> dict[str, object]:
        boundary_vertex_count = len(
            {vertex for edge in self.boundary_edges for vertex in edge}
        )
        implied_genus = None
        if len(self.nonmanifold_edges) == 0:
            genus_numerator = 2 - self.boundary_loop_count - self.euler_characteristic
            if genus_numerator >= 0 and genus_numerator % 2 == 0:
                implied_genus = genus_numerator // 2
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "edge_count": self.edge_count,
            "nonmanifold_edge_count": len(self.nonmanifold_edges),
            "nonmanifold_edges": [list(edge) for edge in self.nonmanifold_edges],
            "orientation_conflict_edge_count": len(self.orientation_conflict_edges),
            "orientation_conflict_edges": [
                list(edge) for edge in self.orientation_conflict_edges
            ],
            "boundary_edge_count": len(self.boundary_edges),
            "boundary_vertex_count": boundary_vertex_count,
            "boundary_loop_count": self.boundary_loop_count,
            "boundary_component_count": self.boundary_component_count,
            "open_boundary_chain_count": self.open_boundary_chain_count,
            "degenerate_face_count": int(np.count_nonzero(self.degenerate_mask)),
            "degenerate_faces": [
                int(index) for index in np.flatnonzero(self.degenerate_mask)
            ],
            "isolated_vertex_count": len(self.isolated_vertices),
            "isolated_vertices": list(self.isolated_vertices),
            "euler_characteristic": self.euler_characteristic,
            "euler_formula": "V - E + F",
            "implied_genus_when_manifold": implied_genus,
            "face_adjacency_count": int(
                sum(len(neighbors) for neighbors in self.face_adjacency) // 2
            ),
            "rescan_matches": self.rescan_matches,
        }


def _boundary_loop_stats(
    boundary_edges: tuple[Edge, ...]
) -> tuple[int, int, int]:
    if not boundary_edges:
        return 0, 0, 0
    neighbors: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    remaining = set(neighbors)
    components = 0
    loops = 0
    open_chains = 0
    while remaining:
        components += 1
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        component: list[int] = []
        while queue:
            vertex = queue.popleft()
            component.append(vertex)
            for neighbor in sorted(neighbors[vertex]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        if all(len(neighbors[vertex]) == 2 for vertex in component):
            loops += 1
        else:
            open_chains += 1
    return loops, components, open_chains


def _naive_rescan(
    faces: np.ndarray,
    degenerate_mask: np.ndarray,
    vertex_count: int,
) -> tuple[int, int, int, int, int, int]:
    edge_incidence: dict[Edge, int] = {}
    valid_edge_incidence: dict[Edge, int] = {}
    edge_directions: dict[Edge, list[int]] = {}
    incident_vertices: set[int] = set()
    for index, face in enumerate(faces.tolist()):
        incident_vertices.update(face)
        edges = set(face_edges(face))
        for edge in edges:
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
            if not bool(degenerate_mask[index]):
                valid_edge_incidence[edge] = valid_edge_incidence.get(edge, 0) + 1
                edge_directions.setdefault(edge, []).append(
                    _face_orientation_for_edge(face, edge)
                )
    nonmanifold = sum(1 for count in valid_edge_incidence.values() if count > 2)
    # Boundary statistics use only non-degenerate faces, matching the indexed pass.
    boundary_edges = tuple(
        sorted(edge for edge, count in valid_edge_incidence.items() if count == 1)
    )
    boundary_loops, _, _ = _boundary_loop_stats(boundary_edges)
    orientation_conflicts = sum(
        1
        for directions in edge_directions.values()
        if len(directions) == 2 and directions[0] == directions[1]
    )
    return (
        len(edge_incidence),
        nonmanifold,
        len(boundary_edges),
        boundary_loops,
        vertex_count - len(incident_vertices),
        orientation_conflicts,
    )


def _face_orientation_for_edge(face: Iterable[int], edge: Edge) -> int:
    a, b, c = (int(value) for value in face)
    for left, right in ((a, b), (b, c), (c, a)):
        if sorted_edge(left, right) == edge:
            return 1 if (left, right) == edge else -1
    raise ValueError("edge is not incident to face")


def build_topology(
    fixed: np.ndarray,
    faces: np.ndarray,
    scale: int,
    area_tolerance: float,
) -> TopologyData:
    vertex_count = int(fixed.shape[0])
    face_count = int(faces.shape[0])
    mask = degenerate_face_mask(fixed, faces, scale, area_tolerance)
    vertex_faces_sets: list[set[int]] = [set() for _ in range(vertex_count)]
    edge_face_sets: dict[Edge, list[int]] = {}
    edge_set_all: set[Edge] = set()

    for face_index, face in enumerate(faces.tolist()):
        for vertex in face:
            vertex_faces_sets[int(vertex)].add(face_index)
        edges = unique_face_edges(face)
        edge_set_all.update(edges)
        if bool(mask[face_index]):
            continue
        for edge in edges:
            edge_face_sets.setdefault(edge, []).append(face_index)

    edge_face_map: dict[Edge, tuple[int, ...]] = {
        edge: tuple(sorted(face_ids)) for edge, face_ids in edge_face_sets.items()
    }
    nonmanifold_edges = tuple(
        sorted(edge for edge, ids in edge_face_map.items() if len(ids) > 2)
    )
    orientation_conflict_edges: list[Edge] = []
    for edge, face_ids in edge_face_map.items():
        if len(face_ids) != 2:
            continue
        left_direction = _face_orientation_for_edge(faces[face_ids[0]].tolist(), edge)
        right_direction = _face_orientation_for_edge(faces[face_ids[1]].tolist(), edge)
        if left_direction == right_direction:
            orientation_conflict_edges.append(edge)
    orientation_conflict_edges.sort()
    boundary_edges = tuple(
        sorted(edge for edge, ids in edge_face_map.items() if len(ids) == 1)
    )
    boundary_loops, boundary_components, open_chains = _boundary_loop_stats(
        boundary_edges
    )

    face_neighbors: list[set[int]] = [set() for _ in range(face_count)]
    for face_ids in edge_face_map.values():
        if len(face_ids) != 2:
            continue
        left, right = face_ids
        face_neighbors[left].add(right)
        face_neighbors[right].add(left)

    neighbor_sets: list[set[int]] = [set() for _ in range(vertex_count)]
    for edge in edge_set_all:
        a, b = edge
        neighbor_sets[a].add(b)
        neighbor_sets[b].add(a)

    isolated = tuple(index for index, ids in enumerate(vertex_faces_sets) if not ids)
    edge_count = len(edge_set_all)
    euler = vertex_count - edge_count + face_count
    naive = _naive_rescan(faces, mask, vertex_count)
    indexed = (
        edge_count,
        len(nonmanifold_edges),
        len(boundary_edges),
        boundary_loops,
        len(isolated),
        len(orientation_conflict_edges),
    )
    return TopologyData(
        vertex_count=vertex_count,
        face_count=face_count,
        edge_count=edge_count,
        euler_characteristic=euler,
        degenerate_mask=mask,
        vertex_faces=tuple(tuple(sorted(ids)) for ids in vertex_faces_sets),
        face_adjacency=tuple(tuple(sorted(ids)) for ids in face_neighbors),
        vertex_neighbors=tuple(tuple(sorted(ids)) for ids in neighbor_sets),
        edge_face_map=edge_face_map,
        boundary_edges=boundary_edges,
        boundary_loop_count=boundary_loops,
        boundary_component_count=boundary_components,
        open_boundary_chain_count=open_chains,
        nonmanifold_edges=nonmanifold_edges,
        orientation_conflict_edges=tuple(orientation_conflict_edges),
        isolated_vertices=isolated,
        rescan_matches=(indexed == naive),
    )


def triangle_signed_volume_numerator(fixed: np.ndarray, faces: np.ndarray) -> int:
    """Exact six-times-volume numerator from authoritative integer coordinates."""

    total = 0
    points = fixed.tolist()
    for face in faces.tolist():
        a = points[int(face[0])]
        b = points[int(face[1])]
        c = points[int(face[2])]
        cross_x = b[1] * c[2] - b[2] * c[1]
        cross_y = b[2] * c[0] - b[0] * c[2]
        cross_z = b[0] * c[1] - b[1] * c[0]
        total += a[0] * cross_x + a[1] * cross_y + a[2] * cross_z
    return int(total)


def geometry_metrics(fixed: np.ndarray, faces: np.ndarray, scale: int) -> dict[str, object]:
    coordinates = fixed_as_float(fixed, scale)
    bbox_min = np.min(coordinates, axis=0) if coordinates.size else np.zeros(3)
    bbox_max = np.max(coordinates, axis=0) if coordinates.size else np.zeros(3)
    points = coordinates[faces]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    volume_numerator = triangle_signed_volume_numerator(fixed, faces)
    volume = volume_numerator / (6.0 * float(scale) ** 3)
    curvature = laplacian_curvature_metric(coordinates, faces)
    return {
        "bbox_min": [float(value) for value in bbox_min],
        "bbox_max": [float(value) for value in bbox_max],
        "bbox_extent": [float(value) for value in (bbox_max - bbox_min)],
        "surface_area": float(np.sum(areas, dtype=np.float64)),
        "signed_volume": float(volume),
        "signed_volume_numerator": volume_numerator,
        "volume_denominator": int(6 * scale**3),
        "mean_uniform_laplacian_magnitude": curvature,
    }


def laplacian_curvature_metric(coordinates: np.ndarray, faces: np.ndarray) -> float:
    neighbors: list[set[int]] = [set() for _ in range(int(coordinates.shape[0]))]
    for face in faces.tolist():
        a, b, c = (int(value) for value in face)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    magnitudes: list[float] = []
    for vertex, adjacent in enumerate(neighbors):
        if not adjacent:
            continue
        mean = np.mean(coordinates[sorted(adjacent)], axis=0, dtype=np.float64)
        magnitudes.append(float(np.linalg.norm(mean - coordinates[vertex])))
    return float(np.mean(magnitudes)) if magnitudes else 0.0

"""Mesh primitives, topology queries and deterministic orientation rules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


GEOMETRY_EPS = 1.0e-12


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("face index outside vertex array")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices contain non-finite values")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.faces))


@dataclass(frozen=True)
class EdgeData:
    edges: np.ndarray
    inverse: np.ndarray
    directions: np.ndarray
    face_ids: np.ndarray
    incident_faces: tuple[tuple[int, ...], ...]

    def face_edge_indices(self, face_id: int) -> tuple[int, int, int]:
        start = face_id * 3
        return (
            int(self.inverse[start]),
            int(self.inverse[start + 1]),
            int(self.inverse[start + 2]),
        )


def half_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.stack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=1,
    ).reshape(-1, 2)
    face_ids = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    return edges, face_ids


def edge_data(faces: np.ndarray) -> EdgeData:
    faces = np.asarray(faces, dtype=np.int64)
    edges, face_ids = half_edges(faces)
    if len(edges) == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return EdgeData(empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), face_ids, ())

    sorted_edges = np.sort(edges, axis=1)
    unique_edges, first, inverse = np.unique(
        sorted_edges, axis=0, return_index=True, return_inverse=True
    )
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1).astype(np.int8)
    same_direction = edges[:, 0] == edges[:, 1]
    directions[same_direction] = 0

    incident: list[list[int]] = [[] for _ in range(len(unique_edges))]
    for half_edge_idx, edge_idx in enumerate(inverse):
        incident[int(edge_idx)].append(int(face_ids[half_edge_idx]))
    packed = tuple(tuple(values) for values in incident)
    return EdgeData(
        edges=unique_edges.astype(np.int64, copy=False),
        inverse=inverse.astype(np.int64, copy=False),
        directions=directions,
        face_ids=face_ids.astype(np.int64, copy=False),
        incident_faces=packed,
    )


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.empty(0, dtype=np.float64)
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    terms = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return float(np.sum(terms) / 6.0)


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    return float(np.sum(triangle_areas(vertices, faces)))


def mesh_components(faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected face components joined by shared undirected edges."""

    faces = np.asarray(faces, dtype=np.int64)
    count = len(faces)
    if count == 0:
        return np.empty(0, dtype=np.int64), 0
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
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    ed = edge_data(faces)
    for incident in ed.incident_faces:
        if len(incident) > 1:
            first = incident[0]
            for other in incident[1:]:
                union(first, other)

    roots = np.array([find(i) for i in range(count)], dtype=np.int64)
    unique_roots = np.unique(roots)
    remap = {int(root): idx for idx, root in enumerate(unique_roots)}
    labels = np.array([remap[int(root)] for root in roots], dtype=np.int64)
    return labels, len(unique_roots)


def edge_diagnostics(faces: np.ndarray) -> dict[str, object]:
    ed = edge_data(faces)
    incidences = np.array([len(values) for values in ed.incident_faces], dtype=np.int64)
    nonmanifold_indices = np.nonzero(incidences > 2)[0]
    boundary_indices = np.nonzero(incidences == 1)[0]

    inconsistent_face = -1
    inconsistent_edge = -1
    inconsistent_count = 0
    for edge_idx in np.nonzero(incidences == 2)[0]:
        values = ed.incident_faces[int(edge_idx)]
        start_a = int(values[0]) * 3
        start_b = int(values[1]) * 3
        dir_a = None
        dir_b = None
        for offset in range(3):
            if int(ed.inverse[start_a + offset]) == int(edge_idx):
                dir_a = int(ed.directions[start_a + offset])
            if int(ed.inverse[start_b + offset]) == int(edge_idx):
                dir_b = int(ed.directions[start_b + offset])
        if dir_a is not None and dir_b is not None and dir_a == dir_b:
            inconsistent_count += 1
            if inconsistent_face < 0:
                inconsistent_face = int(values[1])
                inconsistent_edge = int(edge_idx)

    return {
        "edge_count": int(len(ed.edges)),
        "nonmanifold_edge_count": int(len(nonmanifold_indices)),
        "first_nonmanifold_edge": (
            [int(v) for v in ed.edges[int(nonmanifold_indices[0])]]
            if len(nonmanifold_indices)
            else None
        ),
        "boundary_edge_count": int(len(boundary_indices)),
        "first_boundary_edge": (
            [int(v) for v in ed.edges[int(boundary_indices[0])]]
            if len(boundary_indices)
            else None
        ),
        "inconsistent_winding_edge_count": int(inconsistent_count),
        "first_inconsistent_face": inconsistent_face if inconsistent_face >= 0 else None,
        "first_inconsistent_edge": (
            [int(v) for v in ed.edges[inconsistent_edge]] if inconsistent_edge >= 0 else None
        ),
    }


def orientation_parity(faces: np.ndarray) -> tuple[np.ndarray, bool, dict[str, int] | None]:
    """Return per-face flip parity and whether edge windings can be consistent."""

    faces = np.asarray(faces, dtype=np.int64)
    count = len(faces)
    parity = np.full(count, -1, dtype=np.int8)
    if count == 0:
        return parity, True, None

    ed = edge_data(faces)
    for seed in range(count):
        if parity[seed] >= 0:
            continue
        parity[seed] = 0
        queue: deque[int] = deque([seed])
        while queue:
            face = queue.popleft()
            for edge_idx in ed.face_edge_indices(face):
                incidents = ed.incident_faces[edge_idx]
                if len(incidents) < 2:
                    continue
                own_offset = None
                own_start = face * 3
                for offset in range(3):
                    if int(ed.inverse[own_start + offset]) == edge_idx:
                        own_offset = own_start + offset
                        break
                if own_offset is None:
                    continue
                own_direction = int(ed.directions[own_offset])
                for neighbor in incidents:
                    if neighbor == face:
                        continue
                    neighbor_offset = None
                    neighbor_start = neighbor * 3
                    for offset in range(3):
                        if int(ed.inverse[neighbor_start + offset]) == edge_idx:
                            neighbor_offset = neighbor_start + offset
                            break
                    if neighbor_offset is None:
                        continue
                    required = int(parity[face]) ^ int(
                        own_direction == int(ed.directions[neighbor_offset])
                    )
                    if parity[neighbor] < 0:
                        parity[neighbor] = required
                        queue.append(neighbor)
                    elif int(parity[neighbor]) != required:
                        return parity, False, {
                            "face_a": int(face),
                            "face_b": int(neighbor),
                            "edge_a": int(ed.edges[edge_idx][0]),
                            "edge_b": int(ed.edges[edge_idx][1]),
                        }
    return parity, True, None


def apply_flips(faces: np.ndarray, flip_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(faces, dtype=np.int64).copy()
    mask = np.asarray(flip_mask, dtype=bool)
    result[mask] = result[mask][:, [0, 2, 1]]
    return result


def outward_flip_mask(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Make every orientable component consistently outward-facing."""

    faces = np.asarray(faces, dtype=np.int64)
    parity, consistent, detail = orientation_parity(faces)
    if not consistent:
        raise ValueError(f"faces cannot be made consistently oriented: {detail}")
    flip = parity.astype(bool)
    oriented = apply_flips(faces, flip)
    labels, component_count = mesh_components(oriented)
    for component in range(component_count):
        face_ids = np.nonzero(labels == component)[0]
        component_faces = oriented[face_ids]
        volume = signed_volume(vertices, component_faces)
        if volume < -GEOMETRY_EPS:
            flip[face_ids] = ~flip[face_ids]
    return flip


def orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    return apply_flips(faces, outward_flip_mask(vertices, faces))


def _segment_triangle_intersection(
    p0: np.ndarray, p1: np.ndarray, triangle: np.ndarray, eps: float
) -> bool:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= eps:
        return False
    normal = normal / normal_length
    d0 = float(np.dot(normal, p0 - triangle[0]))
    d1 = float(np.dot(normal, p1 - triangle[0]))
    if (d0 > eps and d1 > eps) or (d0 < -eps and d1 < -eps):
        return False
    if abs(d0) <= eps and abs(d1) <= eps:
        return False

    denominator = d0 - d1
    if abs(denominator) <= eps:
        return False
    t = d0 / denominator
    if t < -eps or t > 1.0 + eps:
        return False
    point = p0 + t * (p1 - p0)
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denominator_bary = dot00 * dot11 - dot01 * dot01
    if abs(denominator_bary) <= eps:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denominator_bary
    v = (dot00 * dot12 - dot01 * dot02) / denominator_bary
    scale_eps = eps * 10.0
    return u >= -scale_eps and v >= -scale_eps and u + v <= 1.0 + scale_eps


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, eps: float) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) <= eps:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denominator
    v = (dot00 * dot12 - dot01 * dot02) / denominator
    return u >= -eps and v >= -eps and u + v <= 1.0 + eps


def _segments_intersect_2d(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, eps: float) -> bool:
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    r = a1 - a0
    s = b1 - b0
    denominator = cross(r, s)
    qp = b0 - a0
    if abs(denominator) <= eps:
        if abs(cross(qp, r)) > eps:
            return False
        rr = float(np.dot(r, r))
        if rr <= eps:
            return False
        t0 = float(np.dot(qp, r) / rr)
        t1 = float(np.dot(b1 - a0, r) / rr)
        low, high = sorted((t0, t1))
        return high >= -eps and low <= 1.0 + eps
    t = cross(qp, s) / denominator
    u = cross(qp, r) / denominator
    return t >= -eps and t <= 1.0 + eps and u >= -eps and u <= 1.0 + eps


def _coplanar_triangles_intersect(t1: np.ndarray, t2: np.ndarray, eps: float) -> bool:
    normal = np.cross(t1[1] - t1[0], t1[2] - t1[0])
    keep = [index for index in range(3) if index != int(np.argmax(np.abs(normal)))]
    p1 = t1[:, keep]
    p2 = t2[:, keep]
    for point in p1:
        if _point_in_triangle_2d(point, p2, eps):
            return True
    for point in p2:
        if _point_in_triangle_2d(point, p1, eps):
            return True
    edges1 = [(p1[0], p1[1]), (p1[1], p1[2]), (p1[2], p1[0])]
    edges2 = [(p2[0], p2[1]), (p2[1], p2[2]), (p2[2], p2[0])]
    return any(
        _segments_intersect_2d(a0, a1, b0, b1, eps)
        for a0, a1 in edges1
        for b0, b1 in edges2
    )


def triangles_intersect(t1: np.ndarray, t2: np.ndarray, eps: float = 1.0e-10) -> bool:
    normal1 = np.cross(t1[1] - t1[0], t1[2] - t1[0])
    normal2 = np.cross(t2[1] - t2[0], t2[2] - t2[0])
    len1 = float(np.linalg.norm(normal1))
    len2 = float(np.linalg.norm(normal2))
    if len1 <= eps or len2 <= eps:
        return False
    normal1 = normal1 / len1
    normal2 = normal2 / len2

    distances2 = (t2 - t1[0]) @ normal1
    if np.all(distances2 > eps) or np.all(distances2 < -eps):
        return False
    distances1 = (t1 - t2[0]) @ normal2
    if np.all(distances1 > eps) or np.all(distances1 < -eps):
        return False

    if float(np.linalg.norm(np.cross(normal1, normal2))) <= eps and abs(float(distances2[0])) <= eps:
        return _coplanar_triangles_intersect(t1, t2, eps)

    for start, end in ((0, 1), (1, 2), (2, 0)):
        if _segment_triangle_intersection(t1[start], t1[end], t2, eps):
            return True
        if _segment_triangle_intersection(t2[start], t2[end], t1, eps):
            return True
    return False


def find_self_intersections(vertices: np.ndarray, faces: np.ndarray, eps: float = 1.0e-10) -> list[tuple[int, int]]:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) < 2:
        return []
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    minima = np.min(triangles, axis=1) - eps
    maxima = np.max(triangles, axis=1) + eps
    areas = triangle_areas(vertices, faces)
    result: list[tuple[int, int]] = []
    for i in range(len(faces) - 1):
        if areas[i] <= eps:
            continue
        upper = len(faces)
        overlap = np.all(
            (minima[i] <= maxima[i + 1 : upper]) & (maxima[i] >= minima[i + 1 : upper]),
            axis=1,
        )
        candidates = np.nonzero(overlap)[0] + i + 1
        if len(candidates) == 0:
            continue
        set_i = set(int(value) for value in faces[i])
        for raw_j in candidates:
            j = int(raw_j)
            if areas[j] <= eps:
                continue
            if set_i.intersection(int(value) for value in faces[j]):
                continue
            if triangles_intersect(triangles[i], triangles[j], eps):
                result.append((i, j))
    return result


def first_duplicate_vertex(vertices: np.ndarray, tolerance: float) -> tuple[int, int] | None:
    """Small deterministic helper used by fixtures and diagnostics."""

    for i in range(len(vertices)):
        delta = np.linalg.norm(vertices[i + 1 :] - vertices[i], axis=1)
        if len(delta) and np.any(delta <= tolerance):
            return i, i + 1 + int(np.nonzero(delta <= tolerance)[0][0])
    return None


def select_faces(faces: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    selected = np.fromiter(indices, dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)[selected]


def component_face_lists(faces: np.ndarray) -> list[list[int]]:
    labels, count = mesh_components(faces)
    return [np.nonzero(labels == component)[0].astype(int).tolist() for component in range(count)]


def canonical_vertex_order(vertices: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(vertices)), vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    return order.astype(np.int64, copy=False)


def make_mesh_for_face_subset(vertices: np.ndarray, faces: np.ndarray, face_indices: Sequence[int]) -> Mesh:
    selected = np.asarray(list(face_indices), dtype=np.int64)
    return Mesh(vertices=np.asarray(vertices), faces=np.asarray(faces)[selected])

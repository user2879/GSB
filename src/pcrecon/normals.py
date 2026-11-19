"""Deterministic normal re-estimation and per-point confidence."""
from __future__ import annotations

import numpy as np

from .errors import NeighborRadiusError
from .spatial import VoxelGrid

ROOM_CENTER = np.array([5.0, 5.0, 5.0], dtype=np.float64)
MIN_NEIGHBORS = 4


def _eigh_descending(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov)
    return vals[::-1], vecs[:, ::-1]


def estimate_normals(
    points: np.ndarray,
    target_indices: np.ndarray,
    radius: float,
    grid: VoxelGrid | None = None,
    *,
    inward_reference: np.ndarray = ROOM_CENTER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-estimate normals for ``target_indices`` only.

    Returns ``(normals_xyz, confidence, neighbour_count)`` arrays sized like
    ``target_indices``.  Eigen-decomposition uses the deterministic LAPACK
    ``eigh`` path; ties are broken by index because neighbourhoods are sorted.
    """
    if radius <= 0.0 or not np.isfinite(radius):
        raise NeighborRadiusError(f"invalid neighbour radius {radius}")
    points = np.ascontiguousarray(points, dtype=np.float64)
    target_indices = np.ascontiguousarray(target_indices, dtype=np.int64)
    grid = grid or VoxelGrid(points, max(radius, 1e-6))

    queries = points[target_indices]
    neighbourhoods = grid.radius_search(queries, radius)

    out_n = np.zeros((target_indices.shape[0], 3), dtype=np.float64)
    out_c = np.zeros(target_indices.shape[0], dtype=np.float64)
    out_k = np.zeros(target_indices.shape[0], dtype=np.int64)

    for row, (idx, qpoint) in enumerate(zip(neighbourhoods, queries)):
        out_k[row] = idx.shape[0]
        if idx.shape[0] < MIN_NEIGHBORS:
            # Degenerate neighbourhood: emit a zero normal and zero confidence
            # rather than inventing orientation.
            continue
        neighbours = points[idx]
        centroid = neighbours.mean(axis=0)
        centered = neighbours - centroid
        cov = (centered.T @ centered) / idx.shape[0]
        eigvals, eigvecs = _eigh_descending(cov)
        normal = eigvecs[:, 2]
        # Orient inward (room interior) deterministically.
        if np.dot(normal, inward_reference - qpoint) < 0.0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        span = eigvals[0] + eigvals[2]
        planarity = (eigvals[1] - eigvals[2]) / span if span > 0.0 else 0.0
        spread = (eigvals[0] - eigvals[2]) / span if span > 0.0 else 0.0
        confidence = float(np.clip(planarity / (spread + 1e-12), 0.0, 1.0))
        out_n[row] = normal
        out_c[row] = confidence
    return out_n, out_c, out_k


def affected_neighbourhood(
    grid: VoxelGrid, changed_points: np.ndarray, radius: float
) -> np.ndarray:
    """All live points within ``radius`` of any changed point (sorted unique)."""
    if changed_points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    hits = grid.radius_search(np.ascontiguousarray(changed_points, dtype=np.float64), radius)
    if not hits:
        return np.empty(0, dtype=np.int64)
    merged = np.unique(np.concatenate([h for h in hits if h.size]))
    return merged


def reestimate_local(
    points: np.ndarray,
    normals: np.ndarray,
    target_indices: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return new normals array (N x 4: xyz + confidence) and an audit record.

    Only rows in ``target_indices`` change; every other row is byte-identical.
    """
    if radius <= 0.0 or not np.isfinite(radius):
        raise NeighborRadiusError(f"invalid neighbour radius {radius}")
    updated = normals.copy()
    grid = VoxelGrid(points, max(radius, 1e-6))
    new_n, new_c, counts = estimate_normals(points, target_indices, radius, grid)
    updated[target_indices, :3] = new_n
    updated[target_indices, 3] = new_c
    audit = {
        "radius": float(radius),
        "reestimated_point_count": int(target_indices.shape[0]),
        "neighbour_count_min": int(counts.min()) if counts.size else 0,
        "neighbour_count_max": int(counts.max()) if counts.size else 0,
        "neighbour_count_mean": float(counts.mean()) if counts.size else 0.0,
        "degenerate_neighbourhoods": int(np.sum(counts < MIN_NEIGHBORS)),
        "target_indices_hash": _index_hash(target_indices),
    }
    return updated, counts, audit


def _index_hash(indices: np.ndarray) -> str:
    from .hashing import digest_array

    return digest_array(np.ascontiguousarray(indices, dtype=np.int64))

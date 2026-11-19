"""Deterministic sphere layouts, angular grids, and clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Return an equal-area Fibonacci sphere, shape ``(n, 3)``.

    The area represented by every point is ``4*pi/n``.  The layout is a pure
    function of ``n`` and the 0-based index, so no RNG state is involved.
    """

    if n_points < 1:
        raise ValueError("n_points must be positive")
    indices = np.arange(int(n_points), dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / float(n_points)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.mod(indices * GOLDEN_ANGLE, 2.0 * np.pi)
    directions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    return normalize_rows(directions)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("cannot normalize a zero vector")
    return vectors / norms


def spherical_weights(n_points: int) -> np.ndarray:
    """Equal-area quadrature weights for the Fibonacci layout."""

    return np.full(int(n_points), 4.0 * np.pi / float(n_points), dtype=np.float64)


def direction_to_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(directions, dtype=np.float64)
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    return azimuth, elevation


def angles_to_directions(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    azimuth = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    elevation = np.radians(np.asarray(elevation_deg, dtype=np.float64))
    cos_elevation = np.cos(elevation)
    return np.column_stack(
        (cos_elevation * np.cos(azimuth), cos_elevation * np.sin(azimuth), np.sin(elevation))
    )


@dataclass(frozen=True)
class EquirectangularGrid:
    directions: np.ndarray
    area_weights: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    shape: tuple[int, int]

    def flatten_shape(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64).reshape(self.shape)


def equirectangular_grid(step_deg: float) -> EquirectangularGrid:
    """Cell-centre equirectangular grid including both poles."""

    step = float(step_deg)
    if step <= 0.0 or step > 45.0:
        raise ValueError("step_deg must be in (0, 45]")
    n_lat = int(round(180.0 / step))
    n_lon = int(round(360.0 / step))
    elevation = -90.0 + step * (np.arange(n_lat, dtype=np.float64) + 0.5)
    azimuth = -180.0 + step * (np.arange(n_lon, dtype=np.float64) + 0.5)
    lon_grid, lat_grid = np.meshgrid(azimuth, elevation)
    directions = angles_to_directions(lon_grid.ravel(), lat_grid.ravel())
    solid_angle = np.radians(step) ** 2 * np.cos(np.radians(lat_grid.ravel()))
    solid_angle = np.maximum(solid_angle, 0.0)
    solid_angle = solid_angle / solid_angle.sum() * (4.0 * np.pi)
    return EquirectangularGrid(
        directions=directions,
        area_weights=solid_angle,
        azimuth_deg=lon_grid.ravel(),
        elevation_deg=lat_grid.ravel(),
        shape=(n_lat, n_lon),
    )


def nearest_indices(query: np.ndarray, catalogue: np.ndarray) -> np.ndarray:
    """Nearest unit-vector index by maximum cosine similarity."""

    query = np.asarray(query, dtype=np.float64)
    catalogue = np.asarray(catalogue, dtype=np.float64)
    return np.argmax(query @ catalogue.T, axis=1)


def farthest_point_representatives(
    directions: np.ndarray,
    priorities: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministic weighted farthest-point representative selection."""

    directions = np.asarray(directions, dtype=np.float64)
    n = len(directions)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    count = min(max(int(count), 1), n)
    priorities = np.asarray(priorities, dtype=np.float64)
    order = np.lexsort((np.arange(n), -priorities))
    selected = [int(order[0])]
    min_distance = 1.0 - directions @ directions[selected[0]]
    for _ in range(1, count):
        candidate = int(np.argmax(min_distance))
        if min_distance[candidate] <= 0.0:
            break
        selected.append(candidate)
        min_distance = np.minimum(min_distance, 1.0 - directions @ directions[candidate])
    return np.asarray(selected, dtype=np.int64)


def cluster_directions(
    directions: np.ndarray,
    priorities: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster every direction exactly once using deterministic representatives.

    Returns ``(representative_indices, assignment)``.  Assignment always
    covers all input directions, including directions that will later be
    excluded by the confidence policy.
    """

    directions = np.asarray(directions, dtype=np.float64)
    priorities = np.asarray(priorities, dtype=np.float64)
    representative_indices = farthest_point_representatives(directions, priorities, cluster_count)
    if len(representative_indices) == 0:
        return representative_indices, np.empty(0, dtype=np.int64)
    similarities = directions @ directions[representative_indices].T
    assignment = np.argmax(similarities, axis=1).astype(np.int64)
    return representative_indices, assignment


def area_weighted_normalize(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    field = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    denominator = float(np.sum(field * weights))
    if denominator <= 0.0:
        return np.zeros_like(field)
    return field / denominator


def energy_centroid(directions: np.ndarray, field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vector = np.sum((np.asarray(field)[:, None] * np.asarray(weights)[:, None]) * directions, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def angular_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


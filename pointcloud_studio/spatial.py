"""Voxel-grid radius lookup used by normal re-estimation and verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelGrid:
    cell_size: float
    origin: np.ndarray
    coordinates: np.ndarray
    sorted_indices: np.ndarray
    starts: np.ndarray
    keys: np.ndarray
    ranges: dict[tuple[int, int, int], tuple[int, int]]

    @classmethod
    def build(cls, points: np.ndarray, cell_size: float) -> "VoxelGrid":
        source = np.asarray(points, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive")
        origin = source.min(axis=0) - cell_size if source.size else np.zeros(3, dtype=np.float64)
        coordinates = np.floor((source - origin) / cell_size).astype(np.int64)
        order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
        sorted_coordinates = coordinates[order]
        if sorted_coordinates.shape[0]:
            change = np.any(np.diff(sorted_coordinates, axis=0) != 0, axis=1)
            starts = np.concatenate(([0], np.flatnonzero(change) + 1))
            keys = sorted_coordinates[starts]
        else:
            starts = np.zeros(0, dtype=np.int64)
            keys = np.zeros((0, 3), dtype=np.int64)
        ranges = {
            (int(key[0]), int(key[1]), int(key[2])): (
                int(starts[index]),
                int(starts[index + 1]) if index + 1 < starts.size else int(order.size),
            )
            for index, key in enumerate(keys)
        }
        return cls(
            cell_size=float(cell_size),
            origin=origin,
            coordinates=coordinates,
            sorted_indices=order.astype(np.int64, copy=False),
            starts=starts.astype(np.int64, copy=False),
            keys=keys.astype(np.int64, copy=False),
            ranges=ranges,
        )

    def cell_of(self, point: np.ndarray) -> tuple[int, int, int]:
        cell = np.floor((np.asarray(point, dtype=np.float64) - self.origin) / self.cell_size).astype(np.int64)
        return int(cell[0]), int(cell[1]), int(cell[2])

    def query_radius(self, points: np.ndarray, point: np.ndarray, radius: float) -> np.ndarray:
        center_cell = self.cell_of(point)
        chunks: list[np.ndarray] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    span = self.ranges.get((center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz))
                    if span is not None:
                        chunks.append(self.sorted_indices[span[0] : span[1]])
        if not chunks:
            return np.zeros(0, dtype=np.int64)
        candidates = np.concatenate(chunks)
        delta = np.asarray(points, dtype=np.float64)[candidates] - np.asarray(point, dtype=np.float64)
        distances_squared = np.einsum("ij,ij->i", delta, delta)
        return candidates[distances_squared <= float(radius) ** 2]


def select_sphere(points: np.ndarray, center: object, radius: float) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    delta = source - origin
    return np.flatnonzero(np.einsum("ij,ij->i", delta, delta) <= float(radius) ** 2)


def select_box(points: np.ndarray, bounds_min: object, bounds_max: object) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    inside = np.all((source >= lower) & (source <= upper), axis=1)
    return np.flatnonzero(inside)


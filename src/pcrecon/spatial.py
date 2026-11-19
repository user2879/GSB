"""Voxel-grid spatial index with deterministic iteration order."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .errors import NeighborRadiusError


class VoxelGrid:
    """Hash-grid spatial index keyed by integer voxel coordinates.

    Iteration over voxels and points is always sorted, so neighbour searches
    return index arrays in ascending order regardless of insertion order.
    """

    def __init__(self, points: np.ndarray, voxel_size: float):
        if voxel_size <= 0.0 or not np.isfinite(voxel_size):
            raise NeighborRadiusError(f"invalid voxel size {voxel_size}")
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.voxel_size = float(voxel_size)
        self.keys = np.floor(self.points / self.voxel_size).astype(np.int64)
        self._cells: dict[tuple[int, int, int], np.ndarray] = {}
        if self.points.shape[0]:
            order = np.lexsort(
                (np.arange(self.points.shape[0]), self.keys[:, 2], self.keys[:, 1], self.keys[:, 0])
            )
            sorted_keys = self.keys[order]
            # Find contiguous cell boundaries deterministically.
            cell_id = self._linear_keys(sorted_keys)
            change = np.flatnonzero(np.diff(cell_id)) + 1
            starts = np.concatenate(([0], change))
            ends = np.concatenate((change, [len(order)]))
            for s, e in zip(starts, ends):
                key = tuple(int(v) for v in sorted_keys[s])
                self._cells[key] = np.sort(order[s:e])

    @staticmethod
    def _linear_keys(keys: np.ndarray) -> np.ndarray:
        # Offset to non-negative space then pack; collision-safe because we
        # still store the true 3-D tuple as the dict key.
        shifted = keys - keys.min(axis=0)
        span = shifted.max(axis=0) + 1
        return (
            shifted[:, 0] * span[1] * span[2]
            + shifted[:, 1] * span[2]
            + shifted[:, 2]
        )

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    def cell_keys(self) -> list[tuple[int, int, int]]:
        return sorted(self._cells)

    def _neighbour_cells(self, key: tuple[int, int, int], radius: float) -> Iterable[tuple[int, int, int]]:
        reach = int(np.ceil(radius / self.voxel_size))
        kx, ky, kz = key
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    yield (kx + dx, ky + dy, kz + dz)

    def radius_search(self, query: np.ndarray, radius: float) -> list[np.ndarray]:
        if radius <= 0.0 or not np.isfinite(radius):
            raise NeighborRadiusError(f"invalid neighbour radius {radius}")
        query = np.ascontiguousarray(query, dtype=np.float64)
        results: list[np.ndarray] = []
        r2 = radius * radius
        for q in query:
            center_key = tuple(int(v) for v in np.floor(q / self.voxel_size))
            candidates: list[np.ndarray] = []
            for cell in self._neighbour_cells(center_key, radius):
                if cell in self._cells:
                    candidates.append(self._cells[cell])
            if candidates:
                idx = np.concatenate(candidates)
                idx.sort()
                diff = self.points[idx] - q
                mask = np.einsum("ij,ij->i", diff, diff) <= r2
                results.append(idx[mask])
            else:
                results.append(np.empty(0, dtype=np.int64))
        return results

    def points_within_region(self, region_min: np.ndarray, region_max: np.ndarray) -> np.ndarray:
        lo = np.asarray(region_min, dtype=np.float64)
        hi = np.asarray(region_max, dtype=np.float64)
        if np.any(hi < lo):
            raise NeighborRadiusError("region_max must be >= region_min")
        mask = np.all((self.points >= lo) & (self.points <= hi), axis=1)
        return np.flatnonzero(mask)

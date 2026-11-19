"""Quadtree blocking over the heightfield.

Conventions:
- depth 0 is the root block covering the whole terrain; leaves sit at
  depth == max_depth and cover BASE_CELLS x BASE_CELLS level-0 cells.
- a block at depth d spans 2**d_per side blocks; block (d, bx, by) covers
  cells [bx*span, (bx+1)*span) with span = (N-1) >> (max_depth - d) ... in
  cells: span = BASE_CELLS << (max_depth - d).
- per-block min/max elevation (uint16) is precomputed bottom-up so LOD error
  bounds can use the block elevation range without touching the heightfield.
"""

from __future__ import annotations

import numpy as np

from .errors import E_LOD_LEVEL, TerrainError
from .heightfield import log2_cells

BASE_CELLS = 32          # leaf block footprint in level-0 cells
MAX_DENSITY = 5          # density g in [0..5]; verts = 2**(5-g)+1, cells = 32>>g


def density_cells(g: int) -> int:
    if not 0 <= g <= MAX_DENSITY:
        raise TerrainError(E_LOD_LEVEL, f"density {g} out of [0, {MAX_DENSITY}]", density=g)
    return BASE_CELLS >> g


def density_verts(g: int) -> int:
    return density_cells(g) + 1


class Quadtree:
    def __init__(self, height0: np.ndarray):
        self.size = height0.shape[0]
        self.n = log2_cells(self.size)
        self.cells = self.size - 1
        if self.cells % BASE_CELLS != 0:
            raise TerrainError(
                E_LOD_LEVEL,
                f"terrain cells {self.cells} not divisible by BASE_CELLS={BASE_CELLS}",
            )
        self.max_depth = self.n - 5  # log2(BASE_CELLS) == 5
        self.height0 = height0
        self._min_maps, self._max_maps = self._build_range_maps(height0)

    def _build_range_maps(self, h: np.ndarray):
        c = self.cells
        corner_max = np.maximum(h[:-1, :-1], h[1:, :-1])
        np.maximum(corner_max, h[:-1, 1:], out=corner_max)
        np.maximum(corner_max, h[1:, 1:], out=corner_max)
        corner_min = np.minimum(h[:-1, :-1], h[1:, :-1])
        np.minimum(corner_min, h[:-1, 1:], out=corner_min)
        np.minimum(corner_min, h[1:, 1:], out=corner_min)
        nb = c // BASE_CELLS
        max_map = corner_max.reshape(nb, BASE_CELLS, nb, BASE_CELLS).max(axis=(1, 3))
        min_map = corner_min.reshape(nb, BASE_CELLS, nb, BASE_CELLS).min(axis=(1, 3))
        max_maps, min_maps = {self.max_depth: max_map}, {self.max_depth: min_map}
        for d in range(self.max_depth - 1, -1, -1):
            m = max_maps[d + 1]
            max_maps[d] = m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).max(axis=(1, 3))
            m = min_maps[d + 1]
            min_maps[d] = m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).min(axis=(1, 3))
        return min_maps, max_maps

    def validate_block(self, depth: int, bx: int, by: int) -> None:
        limit = 1 << depth
        if not (0 <= depth <= self.max_depth and 0 <= bx < limit and 0 <= by < limit):
            raise TerrainError(
                E_LOD_LEVEL,
                f"block ({depth}, {bx}, {by}) out of range (max_depth={self.max_depth})",
                block=(depth, bx, by),
            )

    def block_cells(self, depth: int) -> int:
        if not 0 <= depth <= self.max_depth:
            raise TerrainError(E_LOD_LEVEL, f"depth {depth} out of range", depth=depth)
        return BASE_CELLS << (self.max_depth - depth)

    def block_range(self, depth: int, bx: int, by: int) -> tuple[int, int]:
        self.validate_block(depth, bx, by)
        return (
            int(self._min_maps[depth][by, bx]),
            int(self._max_maps[depth][by, bx]),
        )

    def block_origin_cells(self, depth: int, bx: int, by: int) -> tuple[int, int]:
        span = self.block_cells(depth)
        return bx * span, by * span

    def children(self, depth: int, bx: int, by: int):
        self.validate_block(depth, bx, by)
        if depth >= self.max_depth:
            raise TerrainError(
                E_LOD_LEVEL, "leaf block has no children", block=(depth, bx, by)
            )
        d = depth + 1
        return [(d, 2 * bx + dx, 2 * by + dy) for dy in (0, 1) for dx in (0, 1)]

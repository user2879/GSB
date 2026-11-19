"""Normal evaluation pipeline.

Every vertex normal - interior or seam - is evaluated from the SAME level-0
heightfield with the SAME central-difference operator, so two blocks sharing
a boundary vertex produce bit-identical normals there. Normals never come
from per-block local grids, which would cause lighting seams.
"""

from __future__ import annotations

import numpy as np

from .selection import HEIGHT_SCALE, SAMPLE_SPACING


def normal_at(heights: np.ndarray, i: int, j: int) -> tuple:
    """Normal at global vertex (i, j) via central differences (edge-clamped)."""
    n = heights.shape[0]
    ip, im = min(i + 1, n - 1), max(i - 1, 0)
    jp, jm = min(j + 1, n - 1), max(j - 1, 0)
    dzdx = (float(heights[j, ip]) - float(heights[j, im])) * HEIGHT_SCALE / (
        (ip - im) * SAMPLE_SPACING
    )
    dzdy = (float(heights[jp, i]) - float(heights[jm, i])) * HEIGHT_SCALE / (
        (jp - jm) * SAMPLE_SPACING
    )
    v = np.array([-dzdx, -dzdy, 1.0], dtype=np.float64)
    v /= np.linalg.norm(v)
    return (v[0], v[1], v[2])


def normals_grid(heights: np.ndarray) -> np.ndarray:
    """Vectorized normals for the whole heightfield. Returns (n, n, 3) float32."""
    h = heights.astype(np.float64) * HEIGHT_SCALE
    dzdx = np.zeros_like(h)
    dzdy = np.zeros_like(h)
    dzdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2.0 * SAMPLE_SPACING)
    dzdx[:, 0] = (h[:, 1] - h[:, 0]) / SAMPLE_SPACING
    dzdx[:, -1] = (h[:, -1] - h[:, -2]) / SAMPLE_SPACING
    dzdy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2.0 * SAMPLE_SPACING)
    dzdy[0, :] = (h[1, :] - h[0, :]) / SAMPLE_SPACING
    dzdy[-1, :] = (h[-1, :] - h[-2, :]) / SAMPLE_SPACING
    nx, ny, nz = -dzdx, -dzdy, np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return out.astype(np.float32)

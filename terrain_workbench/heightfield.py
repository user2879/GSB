"""Deterministic procedural heightfield generation (uint16).

Noise model: seeded value noise (PCG64) with bilinear octave upsampling plus
a fixed set of gaussian bumps derived from the same seed. All randomness comes
from numpy's PCG64 bit generator, which is reproducible across runs and
platforms for integer draws. Given the same (size, seed, hmin, hmax) the
output array is byte-identical every time.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .errors import E_HEIGHT_RANGE, TerrainError

VALID_SIZES = (257, 1025, 4097)  # 2^8+1, 2^10+1, 2^12+1
UINT16_MAX = 65535


def validate_elevation_range(hmin: int, hmax: int) -> None:
    if not (0 <= hmin < hmax <= UINT16_MAX):
        raise TerrainError(
            E_HEIGHT_RANGE,
            f"illegal elevation range [{hmin}, {hmax}]; need 0 <= hmin < hmax <= {UINT16_MAX}",
            hmin=hmin,
            hmax=hmax,
        )


def log2_cells(size: int) -> int:
    """Return n for size = 2**n + 1, validating the shape."""
    cells = size - 1
    if cells <= 0 or (cells & (cells - 1)) != 0:
        raise TerrainError(
            E_HEIGHT_RANGE,
            f"heightfield size must be 2**n + 1, got {size}",
            size=size,
        )
    return cells.bit_length() - 1


def bilinear_upsample(grid: np.ndarray, out_size: int) -> np.ndarray:
    """Upsample a (m+1, m+1) float grid to (out_size, out_size) bilinearly.

    Deterministic: pure numpy float64 indexing, no scipy.
    """
    m = grid.shape[0] - 1
    n = out_size - 1
    pos = np.arange(out_size, dtype=np.float64) * (m / n)
    i0 = np.minimum(pos.astype(np.int64), m - 1) if m >= 1 else np.zeros(out_size, np.int64)
    i1 = i0 + 1
    t = pos - i0
    rows0 = grid[i0]
    rows1 = grid[i1]
    # interpolate along x for the two row-slices, then along y
    a = rows0[:, i0] * (1.0 - t)[None, :] + rows0[:, i1] * t[None, :]
    b = rows1[:, i0] * (1.0 - t)[None, :] + rows1[:, i1] * t[None, :]
    return a * (1.0 - t)[:, None] + b * t[:, None]


def _gaussian_bumps(size: int, rng: np.random.Generator, count: int = 6) -> np.ndarray:
    n = size - 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / n
    out = np.zeros((size, size), dtype=np.float64)
    for _ in range(count):
        cx, cy = rng.random(2)
        sigma = 0.08 + 0.15 * rng.random()
        amp = 0.25 + 0.55 * rng.random()
        out += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)))
    return out


def generate_heightfield(size: int, seed: int, hmin: int = 2000,
                         hmax: int | None = None) -> np.ndarray:
    """Generate a deterministic uint16 heightfield of shape (size, size).

    Default relief scales with grid size (8 height units per sample) so all
    three tiers have comparable slopes in world units.
    """
    if hmax is None:
        hmax = hmin + 8 * size
    validate_elevation_range(hmin, hmax)
    n = log2_cells(size)
    rng = np.random.Generator(np.random.PCG64(seed))

    accum = np.zeros((size, size), dtype=np.float64)
    amp_total = 0.0
    # octaves: 4, 8, ... cells up to the grid resolution
    cells = 4
    amp = 1.0
    while cells <= (1 << n):
        octave = rng.integers(0, 1 << 32, size=(cells + 1, cells + 1), dtype=np.uint32)
        octave = octave.astype(np.float64) / 4294967296.0
        accum += amp * bilinear_upsample(octave, size)
        amp_total += amp
        cells *= 2
        amp *= 0.5
    accum /= amp_total

    bumps = _gaussian_bumps(size, rng)
    accum = 0.72 * accum + 0.28 * (bumps / bumps.max() if bumps.max() > 0 else bumps)

    scaled = np.floor(hmin + (hmax - hmin) * accum + 0.5)
    return np.clip(scaled, hmin, hmax).astype(np.uint16)


def heightfield_hash(heights: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(heights).tobytes()).hexdigest()

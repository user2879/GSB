"""Deterministic value noise derived only from pixel coordinates and a
layer fingerprint.  No runtime RNG is ever consulted; re-running with the same
inputs always yields bit-identical noise fields.
"""
from __future__ import annotations

import numpy as np

_MASK64 = np.uint64((1 << 64) - 1)


def _splitmix64(z: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 over uint64 inputs -> uint64 hash words."""
    z = z.astype(np.uint64)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
    return z ^ (z >> np.uint64(31))


def _lattice_hash(ix: np.ndarray, iy: np.ndarray, fingerprint: int) -> np.ndarray:
    base = np.uint64(int(fingerprint))
    h = _splitmix64((base ^ np.uint64(ix.astype(np.int64).view(np.uint64))) & _MASK64)
    h = _splitmix64((h ^ np.uint64(iy.astype(np.int64).view(np.uint64))) & _MASK64)
    return h


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise_field(width: int, height: int, frequency: float, fingerprint: int) -> np.ndarray:
    """Smooth value noise in ``[-1, 1]``, shape ``(height, width)``.

    Lattice corner hashes come from :func:`_lattice_hash`; bilinear
    interpolation with smoothstep easing produces the continuous field.  The
    traversal is row-major (``y`` outer, ``x`` inner) as required by the
    pipeline contract.
    """
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)  # shape (H, W), row-major
    fx = gx * float(frequency)
    fy = gy * float(frequency)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    tx = fx - np.floor(fx)
    ty = fy - np.floor(fy)
    sx = _smoothstep(tx)
    sy = _smoothstep(ty)

    def corner(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        h = _lattice_hash(ix, iy, fingerprint)
        # map full uint64 word uniformly into [-1, 1]
        return (h.astype(np.float64) / float(1 << 64)) * 2.0 - 1.0

    c00 = corner(x0, y0)
    c10 = corner(x1, y0)
    c01 = corner(x0, y1)
    c11 = corner(x1, y1)
    top = c00 + (c10 - c00) * sx
    bottom = c01 + (c11 - c01) * sx
    field = top + (bottom - top) * sy
    return field.astype(np.float64)


def pattern_control_points(seed: int, width: int, height: int, count: int = 4) -> np.ndarray:
    """Deterministic control-point grid (x, y, radius, strength) for the
    fixed-seed base texture generator.
    """
    rng_words = _splitmix64(
        np.arange(1, count * 4 + 1, dtype=np.uint64) + np.uint64(int(seed))
    ).astype(np.float64) / float(1 << 64)
    points = np.zeros((count, 4), dtype=np.float64)
    for i in range(count):
        points[i, 0] = rng_words[i * 4] * width
        points[i, 1] = rng_words[i * 4 + 1] * height
        points[i, 2] = 0.08 + 0.12 * rng_words[i * 4 + 2]
        points[i, 3] = rng_words[i * 4 + 3]
    return points

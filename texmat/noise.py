"""Coordinate-derived deterministic noise.

Noise at a pixel is a pure function of (x, y, layer fingerprint, seed); no RNG
sampling is involved anywhere, so identical inputs reproduce identical pixels.
"""

from __future__ import annotations

import numpy as np

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix64(value: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 mixing step on uint64 arrays."""
    with np.errstate(over="ignore"):
        value = (value + np.uint64(0x9E3779B97F4A7C15)) & _MASK64
        value = ((value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK64
        value = ((value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK64
        return value ^ (value >> np.uint64(31))


def _hash_pixels(coords: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        h = (coords + np.uint64(seed & 0xFFFFFFFFFFFFFFFF)) & _MASK64
        h = (h ^ (h >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD) & _MASK64
        h = (h ^ (h >> np.uint64(33))) * np.uint64(0xC4CEB9FE1A85EC53) & _MASK64
        h = h ^ (h >> np.uint64(33))
        return _mix64(h)


def pixel_hash_grid(width: int, height: int, fingerprint: int, seed: int) -> np.ndarray:
    """Return a per-pixel uint64 hash grid in row-major (C) order."""
    ys, xs = np.meshgrid(np.arange(height, dtype=np.uint64), np.arange(width, dtype=np.uint64), indexing="ij")
    coords = (xs * np.uint64(0x9E3779B1) + ys * np.uint64(0x85EBCA77)
              + np.uint64(fingerprint & 0xFFFFFFFFFFFFFFFF)) & _MASK64
    return _hash_pixels(coords, seed).reshape(height, width)


def noise_grid(width: int, height: int, fingerprint: int, seed: int) -> np.ndarray:
    """Deterministic noise in the closed range [0, 1], row-major float64."""
    hashed = pixel_hash_grid(width, height, fingerprint, seed)
    return (hashed.astype(np.float64) / float(0xFFFFFFFFFFFFFFFF))


def value_noise(width: int, height: int, frequency: int, fingerprint: int, seed: int) -> np.ndarray:
    """Lattice value noise: hashed lattice points at ``frequency`` per axis,
    bilinear interpolation in continuous pixel coordinates, output [0, 1]."""
    if frequency < 1:
        raise ValueError("frequency must be a positive integer")
    lat_y, lat_x = np.meshgrid(np.arange(frequency + 1, dtype=np.uint64),
                               np.arange(frequency + 1, dtype=np.uint64), indexing="ij")
    coords = (lat_x * np.uint64(0x9E3779B1) + lat_y * np.uint64(0x85EBCA77)
              + np.uint64(fingerprint)) & _MASK64
    with np.errstate(over="ignore"):
        hashed = _hash_pixels(coords, seed)
    vals_lat = hashed.astype(np.float64) / float(0xFFFFFFFFFFFFFFFF)
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float64),
                         np.arange(width, dtype=np.float64), indexing="ij")
    gx = xs * float(frequency) / max(width, 1)
    gy = ys * float(frequency) / max(height, 1)
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.minimum(x0 + 1, frequency)
    y1 = np.minimum(y0 + 1, frequency)
    x0 = np.minimum(x0, frequency - 1)
    y0 = np.minimum(y0, frequency - 1)
    tx = gx - np.floor(gx)
    ty = gy - np.floor(gy)
    top = vals_lat[y0, x0] * (1.0 - tx) + vals_lat[y0, x1] * tx
    bottom = vals_lat[y1, x0] * (1.0 - tx) + vals_lat[y1, x1] * tx
    return (top * (1.0 - ty) + bottom * ty).astype(np.float64)

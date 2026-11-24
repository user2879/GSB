"""Deterministic coordinate-derived procedural value noise."""

from __future__ import annotations

import hashlib
import math

import numpy as np


_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_X_SALT = np.uint64(0xD6E8FEB86659FD93)
_Y_SALT = np.uint64(0xA0761D6478BD642F)


def splitmix64(value: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        value = (np.asarray(value, dtype=np.uint64) + _GOLDEN) & _MASK
        value = ((value ^ (value >> np.uint64(30))) * _MIX1) & _MASK
        value = ((value ^ (value >> np.uint64(27))) * _MIX2) & _MASK
        value = (value ^ (value >> np.uint64(31))) & _MASK
    return value


def layer_seed(seed: int, layer_index: int, payload: dict[str, object]) -> np.uint64:
    import json

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha256(f"{seed}:{layer_index}:".encode("ascii") + encoded).digest()
    return np.uint64(int.from_bytes(digest[:8], "big"))


def _hash_grid(seed: np.uint64, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x64 = np.asarray(x, dtype=np.int64).astype(np.uint64)
        y64 = np.asarray(y, dtype=np.int64).astype(np.uint64)
        mixed = splitmix64(
            seed
            ^ splitmix64(x64 * _X_SALT)
            ^ splitmix64(y64 * _Y_SALT)
        )
    # Use the top 53 bits so the result is exactly representable as float64.
    return (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def value_noise(seed: np.uint64, frequency: float, row_start: int, row_end: int, width: int) -> np.ndarray:
    if row_end <= row_start or width <= 0:
        return np.empty((max(0, row_end - row_start), width), dtype=np.float64)
    ys = np.arange(row_start, row_end, dtype=np.float64) * frequency
    xs = np.arange(width, dtype=np.float64) * frequency
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    sy = fy * fy * (3.0 - 2.0 * fy)
    sx = fx * fx * (3.0 - 2.0 * fx)
    yy0 = y0[:, None]
    yy1 = yy0 + 1
    xx0 = x0[None, :]
    xx1 = xx0 + 1
    v00 = _hash_grid(seed, xx0, yy0)
    v10 = _hash_grid(seed, xx1, yy0)
    v01 = _hash_grid(seed, xx0, yy1)
    v11 = _hash_grid(seed, xx1, yy1)
    top = v00 * (1.0 - sx)[None, :] + v10 * sx[None, :]
    bottom = v01 * (1.0 - sx)[None, :] + v11 * sx[None, :]
    value = top * (1.0 - sy)[:, None] + bottom * sy[:, None]
    return value * 2.0 - 1.0


def layer_payload(
    *,
    seed: int,
    index: int,
    texture_key: str,
    frequency: float,
    amplitude: float,
    resolution: int,
) -> dict[str, object]:
    return {
        "kind": "noise_layer",
        "version": 1,
        "seed": seed,
        "index": index,
        "texture_key": texture_key,
        "frequency": float(frequency).hex(),
        "amplitude": float(amplitude).hex(),
        "resolution": resolution,
    }


def derive_layer_seed(payload: dict[str, object]) -> np.uint64:
    return layer_seed(int(payload["seed"]), int(payload["index"]), payload)


def finite_or_raise(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

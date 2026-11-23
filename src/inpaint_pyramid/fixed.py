"""Integer-only fixed point helpers.

All stored image samples are int64 with ``fraction_bits`` fractional bits.
Rounding is round-half-up for non-negative image values.
"""

from __future__ import annotations

import numpy as np


def round_div(numerator: np.ndarray | int, denominator: np.ndarray | int) -> np.ndarray | int:
    """Round non-negative integer division half up."""
    return (numerator + denominator // 2) // denominator


def to_fixed(rgb: np.ndarray, fraction_bits: int) -> np.ndarray:
    return np.asarray(rgb, dtype=np.int64) << int(fraction_bits)


def from_fixed_fixed(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    scale = np.int64(1 << fraction_bits)
    return np.asarray((values + scale // 2) // scale, dtype=np.int64)


def from_fixed_u8(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    rgb = from_fixed_fixed(values, fraction_bits)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def to_u8_index(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    return np.clip(from_fixed_fixed(values, fraction_bits), 0, 255).astype(np.int64)


def blend_fixed(a: np.ndarray, b: np.ndarray, weight: int, scale: int) -> np.ndarray:
    """Return (1-w)*a + w*b with integer half-up rounding."""
    weight = int(weight)
    return ((scale - weight) * a + weight * b + scale // 2) // scale


def scalar_to_fixed(value: float, fraction_bits: int) -> int:
    scale = 1 << int(fraction_bits)
    return max(0, min(scale, int(round(float(value) * scale))))


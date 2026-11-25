"""Fixed-point arithmetic primitives.

Pixels are represented as signed int64 values in a common fractional domain.
All render-path numeric operations below are integer operations; floating
point is used only while validating and compiling declarative weights.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

ALLOWED_FRAC_BITS = (8, 12, 16, 20)
ALLOWED_QUANT_STEPS = (1, 2, 4, 8, 16)
TIE_RULES = ("ties_to_even", "ties_to_lower", "ties_to_upper")
EQUAL_WEIGHT_RULES = ("lower_layer_index", "upper_layer_index", "first_declared")
BLEND_MODES = ("weighted", "max", "layered_over")
CHANNEL_ORDER = ("diffuse", "roughness", "metallic", "normal")

DECISION_DOWN = np.int8(0)
DECISION_UP = np.int8(1)
DECISION_TIE_DOWN = np.int8(2)
DECISION_TIE_UP = np.int8(3)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_positive_finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number > 0.0


def scale_for(frac_bits: int) -> int:
    return 1 << int(frac_bits)


def clamp_array(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(values, low, high).astype(np.int64, copy=False)


def round_ratio_scalar(numerator: int, denominator: int, tie_rule: str) -> tuple[int, int, str]:
    """Round a non-negative rational value using the configured tie rule."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient, remainder = divmod(int(numerator), int(denominator))
    doubled = remainder * 2
    if doubled > denominator:
        return quotient + 1, remainder, "up"
    if doubled < denominator:
        return quotient, remainder, "down"
    if tie_rule == "ties_to_upper":
        return quotient + 1, remainder, "tie_up"
    if tie_rule == "ties_to_lower":
        return quotient, remainder, "tie_down"
    if quotient % 2:
        return quotient + 1, remainder, "tie_even_up"
    return quotient, remainder, "tie_even_down"


def round_ratio_array(
    numerator: np.ndarray, denominator: int, tie_rule: str
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized non-negative rational rounding.

    Returns ``(rounded, decisions)`` where decisions are one of the four
    DECISION_* int8 constants.
    """

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    num = np.asarray(numerator, dtype=np.int64)
    quotient = np.floor_divide(num, denominator)
    remainder = num - quotient * denominator
    doubled = remainder * 2
    decisions = np.full(num.shape, DECISION_DOWN, dtype=np.int8)
    up_mask = doubled > denominator
    decisions[up_mask] = DECISION_UP
    tie_mask = doubled == denominator
    if tie_rule == "ties_to_upper":
        decisions[tie_mask] = DECISION_TIE_UP
        up_mask = up_mask | tie_mask
    elif tie_mask.any():
        if tie_rule == "ties_to_lower":
            decisions[tie_mask] = DECISION_TIE_DOWN
        else:
            even_up = tie_mask & (quotient % 2 == 1)
            even_down = tie_mask & ~even_up
            decisions[even_up] = DECISION_TIE_UP
            decisions[even_down] = DECISION_TIE_DOWN
            up_mask = up_mask | even_up
    rounded = quotient + up_mask.astype(np.int64)
    return rounded, decisions


def quantize_fixed(
    raw: np.ndarray, step: int, tie_rule: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a fixed-point array to a lattice and retain the residual."""

    raw_array = np.asarray(raw, dtype=np.int64)
    quotient, decisions = round_ratio_array(raw_array, int(step), tie_rule)
    quantized = quotient * int(step)
    residual = raw_array - quantized
    return quantized.astype(np.int64, copy=False), residual.astype(np.int64, copy=False), decisions


def u8_to_fixed(values: np.ndarray, frac_bits: int) -> np.ndarray:
    scale = scale_for(frac_bits)
    return ((np.asarray(values, dtype=np.int64) * scale) + 127) // 255


def fixed_to_u8(values: np.ndarray, frac_bits: int) -> np.ndarray:
    scale = scale_for(frac_bits)
    scaled = np.asarray(values, dtype=np.int64)
    rounded = (scaled * 255 + scale // 2) // scale
    return np.clip(rounded, 0, 255).astype(np.uint8)


def downscale_fixed(
    values: np.ndarray, source_bits: int, target_bits: int
) -> np.ndarray:
    """Round a higher-precision fixed array down to ``target_bits``."""

    if target_bits >= source_bits:
        factor = 1 << (target_bits - source_bits)
        return (np.asarray(values, dtype=np.int64) * factor).astype(np.int64, copy=False)
    ratio = 1 << (source_bits - target_bits)
    array = np.asarray(values, dtype=np.int64)
    return np.floor_divide(array + ratio // 2, ratio).astype(np.int64, copy=False)


def array_digest(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()

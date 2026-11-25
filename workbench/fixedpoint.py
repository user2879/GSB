from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np


def scale_for_bits(fractional_bits: int) -> int:
    return 1 << fractional_bits


def decimal_to_fixed(value: Any, fractional_bits: int) -> int:
    """Convert JSON numeric input exactly enough for cross-platform planning."""
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a decimal number: {value!r}") from exc
    scale = Decimal(1 << fractional_bits)
    return int((decimal_value * scale).to_integral_value(rounding="ROUND_HALF_UP"))


def round_half_up(numerator: np.ndarray | int, denominator: int) -> np.ndarray | int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if isinstance(numerator, int):
        return (numerator * 2 + denominator) // (2 * denominator)
    array = np.asarray(numerator, dtype=np.int64)
    return (array * 2 + denominator) // (2 * denominator)


def fixed_to_byte_array(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    scale = scale_for_bits(fractional_bits)
    rounded = np.floor_divide(
        values.astype(np.int64) * 2 + scale,
        2 * scale,
    )
    return np.clip(rounded, 0, 255).astype(np.uint8)


def byte_to_fixed_array(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    return values.astype(np.int64) * scale_for_bits(fractional_bits)

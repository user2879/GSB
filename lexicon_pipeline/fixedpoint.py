"""Deterministic fixed-point helpers.

The production recurrence uses arbitrary-precision Python integers.  NumPy is
used by the sample generator and by an optional cross-check, never for network
or service access.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Any


ROUNDING_CODES: dict[str, int] = {
    "EXACT": 0,
    "ROUND_HALF_UP": 1,
    "ROUND_HALF_EVEN": 2,
    "ROUND_DOWN": 3,
}
CODE_TO_ROUNDING = {code: name for name, code in ROUNDING_CODES.items()}
_DECIMAL_MODES = {
    "ROUND_HALF_UP": ROUND_HALF_UP,
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
    "ROUND_DOWN": ROUND_DOWN,
    "EXACT": ROUND_HALF_UP,
}


def quantize_int(value: float | int, scale: int, mode: str = "ROUND_HALF_UP") -> int:
    """Quantize a finite decimal input to an integer at ``scale`` units."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    decimal_value = Decimal(str(value)) * Decimal(scale)
    rounding = _DECIMAL_MODES[mode]
    if mode == "EXACT":
        return int(decimal_value)
    return int(decimal_value.to_integral_value(rounding=rounding))


def decoded_float(value_int: int, scale: int) -> float:
    return value_int / scale


def quantization_residual(value: float | int, value_int: int, scale: int) -> float:
    return float(Decimal(str(value)) - Decimal(value_int) / Decimal(scale))


def numpy_integer_sum(values: list[int]) -> Any:
    """Cross-check integer accumulation with NumPy's signed 64-bit arrays."""

    import numpy as np

    array = np.asarray(values, dtype=np.int64)
    return int(array.sum(dtype=np.int64))

"""Deterministic fixed-point integer arithmetic.

Every ratio, lift and interval endpoint is represented internally as a signed
integer scaled by ``10**decimal_width``. Quantization levels (when configured)
further snap values onto an integer grid expressed in *scaled* units. All
rounding is explicit and deterministic (floor / ceil); floating point is used
only inside variance and square-root estimation and is immediately re-quantized.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional


class FixedPoint:
    """Fixed-point context for one declared precision."""

    def __init__(self, decimal_width: int, quantization_step: Optional[str] = None):
        if not isinstance(decimal_width, int) or decimal_width < 0:
            raise ValueError("decimal_width must be a non-negative integer")
        self.width = decimal_width
        self.scale = 10 ** decimal_width
        step_grid = 1
        if quantization_step not in (None, "", 0, "0"):
            step_dec = Decimal(str(quantization_step))
            if step_dec <= 0:
                raise ValueError("quantization_step must be positive when provided")
            step_grid = int((step_dec * Decimal(self.scale)).to_integral_value())
            if step_grid <= 0:
                raise ValueError("quantization_step is smaller than the fixed-point unit")
        self.quantization_step = str(quantization_step) if quantization_step else None
        self.grid = step_grid

    @property
    def precision_label(self) -> str:
        if self.grid > 1:
            return f"fixed<{self.width}>,quant={self.quantization_step}"
        return f"fixed<{self.width}>"

    def ratio(self, numerator: int, denominator: int) -> int:
        """Exact integer floor ratio returned as a scaled integer."""
        if denominator <= 0:
            raise ZeroDivisionError("ratio denominator must be positive")
        return (int(numerator) * self.scale) // int(denominator)

    def from_int_units(self, value: int) -> int:
        return int(value) * self.scale

    def quantize(self, scaled_value: int) -> int:
        """Snap a scaled integer onto the quantization grid (floor toward -inf)."""
        if self.grid <= 1:
            return int(scaled_value)
        return (int(scaled_value) // self.grid) * self.grid

    def floor_scaled(self, float_value: float) -> int:
        return self.quantize(_floor_float(float_value * self.scale))

    def ceil_scaled(self, float_value: float) -> int:
        return self.quantize(_ceil_float(float_value * self.scale))

    def to_text(self, scaled_value: int) -> str:
        """Render a scaled integer with exactly ``width`` decimal digits."""
        value = int(scaled_value)
        sign = "-" if value < 0 else ""
        value = abs(value)
        whole, frac = divmod(value, self.scale)
        if self.width:
            return f"{sign}{whole}.{str(frac).zfill(self.width)}"
        return f"{sign}{whole}"


def _floor_float(x: float) -> int:
    import math

    return int(math.floor(x))


def _ceil_float(x: float) -> int:
    import math

    return int(math.ceil(x))

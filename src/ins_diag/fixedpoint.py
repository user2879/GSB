from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .config import FixedPointConfig
from .util import fingerprint, round_half_away_from_zero


@dataclass(frozen=True)
class FixedPoint:
    fractional_bits: int
    quantization_units: int

    @classmethod
    def from_config(cls, config: FixedPointConfig) -> "FixedPoint":
        return cls(config.fractional_bits, config.quantization_units)

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    def quantize(self, value: float) -> int:
        if not math.isfinite(value):
            raise ValueError("cannot quantize a non-finite value")
        grid_index = round_half_away_from_zero((value * self.scale) / self.quantization_units)
        return grid_index * self.quantization_units

    def quantize_vector(self, values: Iterable[float]) -> list[int]:
        return [self.quantize(float(value)) for value in values]

    def dequantize(self, value: int) -> float:
        return value / self.scale

    def dequantize_vector(self, values: Iterable[int]) -> list[float]:
        return [self.dequantize(int(value)) for value in values]

    def residual_units(self, value: float, quantized: int) -> int:
        return int(round(abs(value * self.scale - quantized)))

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "fractional_bits": self.fractional_bits,
                "quantization_units": self.quantization_units,
            }
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "fractional_bits": self.fractional_bits,
            "quantization_units": self.quantization_units,
            "scale": self.scale,
        }


def rounded_grid_division(total_units: int, count: int, quantum: int) -> int:
    """Round total/count to the nearest multiple of quantum without floats."""
    if count <= 0:
        raise ValueError("count must be positive")
    divisor = count * quantum
    absolute = abs(total_units)
    quotient, remainder = divmod(absolute, divisor)
    if remainder * 2 >= divisor:
        quotient += 1
    result = quotient * quantum
    return result if total_units >= 0 else -result


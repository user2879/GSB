"""Deterministic fixed-point arithmetic used by every numerical stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import BusinessError
from .util import clamp_int, quantize_int, round_div

ALLOWED_FRACTION_BITS = (6, 8, 10, 12)
ALLOWED_STORAGE_BITS = (16, 24, 32)
ALLOWED_QUANT_LEVELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class FixedPoint:
    fraction_bits: int
    storage_bits: int
    quant_level: int

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    @property
    def quant_step(self) -> int:
        return 1 << self.quant_level

    @property
    def low(self) -> int:
        return -(1 << (self.storage_bits - 1))

    @property
    def high(self) -> int:
        return (1 << (self.storage_bits - 1)) - 1

    @property
    def numeric_quant_step(self) -> str:
        if self.quant_level <= self.fraction_bits:
            return f"1/{1 << (self.fraction_bits - self.quant_level)}"
        return str(1 << (self.quant_level - self.fraction_bits))

    def as_dict(self) -> dict[str, int | str]:
        return {
            "fraction_bits": self.fraction_bits,
            "storage_bits": self.storage_bits,
            "quant_level": self.quant_level,
            "scale": self.scale,
            "integer_quant_step": self.quant_step,
            "numeric_quant_step": self.numeric_quant_step,
        }

    def validate(self) -> None:
        if self.fraction_bits not in ALLOWED_FRACTION_BITS:
            raise BusinessError("E_ENUM_FRACTION_BITS", "fixed fraction bits is outside the allowed enum",
                                {"value": self.fraction_bits, "allowed": list(ALLOWED_FRACTION_BITS)})
        if self.storage_bits not in ALLOWED_STORAGE_BITS:
            raise BusinessError("E_ENUM_STORAGE_BITS", "fixed storage bits is outside the allowed enum",
                                {"value": self.storage_bits, "allowed": list(ALLOWED_STORAGE_BITS)})
        if self.quant_level not in ALLOWED_QUANT_LEVELS:
            raise BusinessError("E_ENUM_QUANT_LEVEL", "quantization level is outside the allowed enum",
                                {"value": self.quant_level, "allowed": list(ALLOWED_QUANT_LEVELS)})
        if self.quant_level >= self.storage_bits:
            raise BusinessError("E_FIXED_RANGE", "quantization level must be below storage width")

    def quantize(self, value: int) -> tuple[int, int]:
        return quantize_int(value, self.quant_level)

    def from_real(self, value: float, *, clamp: bool = True) -> int:
        number = float(value) * self.scale
        raw = int(math.floor(number + 0.5)) if number >= 0 else int(math.ceil(number - 0.5))
        quantized, _ = self.quantize(raw)
        return clamp_int(quantized, self.low, self.high) if clamp else quantized

    def from_ratio(self, numerator: int, denominator: int, *, clamp: bool = True) -> int:
        if denominator <= 0:
            raise BusinessError("E_FIXED_DIVIDE", "denominator must be positive")
        raw = round_div(numerator * self.scale, denominator)
        quantized, _ = self.quantize(raw)
        return clamp_int(quantized, self.low, self.high) if clamp else quantized

    def mul(self, left: int, right: int) -> int:
        raw = round_div(left * right, self.scale)
        quantized, _ = self.quantize(raw)
        return clamp_int(quantized, self.low, self.high)

    def add(self, left: int, right: int) -> int:
        raw = left + right
        quantized, _ = self.quantize(raw)
        return clamp_int(quantized, self.low, self.high)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "fraction_bits": self.fraction_bits,
            "storage_bits": self.storage_bits,
            "quant_level": self.quant_level,
        }

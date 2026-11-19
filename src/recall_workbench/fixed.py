"""Deterministic fixed-point arithmetic with half-even rounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_ROUNDING = "half_even"


def round_ratio(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest, with exact ties going to even."""

    if denominator == 0:
        raise ZeroDivisionError("fixed-point denominator must not be zero")
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    numerator = abs(int(numerator))
    denominator = abs(int(denominator))
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def round_half_even(value: float) -> int:
    """Round one finite float to the nearest integer using ties-to-even."""

    if not np.isfinite(value):
        raise ValueError(f"cannot quantize non-finite value: {value!r}")
    return int(np.rint(float(value)))


@dataclass(frozen=True)
class FixedFormat:
    bits: int
    rounding: str = SUPPORTED_ROUNDING

    def __post_init__(self) -> None:
        if self.bits < 1 or self.bits > 30:
            raise ValueError("fixed-point bits must be between 1 and 30")
        if self.rounding != SUPPORTED_ROUNDING:
            raise ValueError(f"unsupported rounding mode: {self.rounding}")

    @property
    def scale(self) -> int:
        return 1 << self.bits

    @property
    def quantum(self) -> float:
        return 1.0 / self.scale

    def quantize(self, value: float) -> int:
        return round_half_even(float(value) * self.scale)

    def quantize_array(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("fixed-point input contains NaN or infinity")
        return np.rint(array * self.scale).astype(np.int64)

    def to_float(self, fixed_value: int) -> float:
        return int(fixed_value) / self.scale

    def multiply(self, left: int, right: int) -> int:
        return round_ratio(int(left) * int(right), self.scale)

    def divide(self, value: int, divisor: int) -> int:
        return round_ratio(int(value) * self.scale, int(divisor))

    def divide_float(self, value: int, divisor: float) -> int:
        return round_ratio(int(value) * self.scale, self.quantize(divisor))

    def clamp(self, value: int, minimum: int, maximum: int) -> int:
        return min(max(int(value), int(minimum)), int(maximum))

    def rescale_to(self, value: int, target: FixedFormat) -> int:
        if target.bits >= self.bits:
            return int(value) << (target.bits - self.bits)
        return round_ratio(int(value), 1 << (self.bits - target.bits))


def sum_fixed(values: Iterable[int]) -> int:
    return int(sum(int(value) for value in values))


def fixed_difference(primary_value: int, primary: FixedFormat, reference_value: int, reference: FixedFormat) -> float:
    """Return |primary-reference| in reference-format score units."""

    if reference.bits < primary.bits:
        raise ValueError("reference format must have at least as many fractional bits")
    aligned = primary.rescale_to(primary_value, reference)
    return abs(aligned - int(reference_value)) / reference.scale

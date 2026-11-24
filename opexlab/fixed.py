"""Integer-only fixed-point primitives used by both operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .util import div_round_half_away, quantize_int_to_step, round_half_away


LN2_MICRO = 693147180559  # ln(2) with 12 decimal places, used as an integer constant.


@dataclass(frozen=True)
class FixedContext:
    fractional_bits: int
    quant_level: int

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def quantization_step(self) -> int:
        return 1 << self.quant_level

    def quantize_float(self, value: float) -> int:
        raw = round_half_away(float(value) * self.scale)
        return quantize_int_to_step(raw, self.quantization_step)

    def quantize_raw(self, value: int) -> int:
        return quantize_int_to_step(value, self.quantization_step)

    def mul(self, left: int, right: int) -> int:
        return div_round_half_away(left * right, self.scale)

    def div(self, numerator: int, denominator: int) -> int:
        if denominator == 0:
            raise ZeroDivisionError("fixed-point division by zero")
        return div_round_half_away(numerator * self.scale, denominator)

    def to_float(self, value: int) -> float:
        return value / self.scale


def _ln2_fixed(fractional_bits: int) -> int:
    scale = 1 << fractional_bits
    return (LN2_MICRO * scale + 500_000_000_000) // 1_000_000_000_000


def exp_negative_fixed(
    z_fixed: int,
    context: FixedContext,
    polynomial_order: int,
) -> int:
    """Approximate exp(-z) for z >= 0 using integer range reduction.

    z is split as ``k*ln(2) + r`` with ``0 <= r < ln(2)``.  ``exp(-r)`` is
    evaluated by a truncated Taylor series using rounded integer
    multiplication, then shifted right by ``k``.  No floating-point operation
    participates in the actual per-element evaluation.
    """
    if z_fixed < 0:
        raise ValueError("exp_negative_fixed requires z >= 0")
    if z_fixed == 0:
        return context.scale
    ln2_fixed = _ln2_fixed(context.fractional_bits)
    if ln2_fixed <= 0:
        return context.scale
    k, remainder = divmod(z_fixed, ln2_fixed)
    if k > 256:
        return 0
    term = context.scale
    total = context.scale
    for index in range(1, polynomial_order + 1):
        term = context.mul(term, remainder)
        term = div_round_half_away(term, index)
        if term == 0 and index > 2:
            break
        if index % 2:
            total -= term
        else:
            total += term
    if total < 0:
        total = 0
    if k == 0:
        return total
    # Round-to-nearest after division by 2**k, with an integer-only guard
    # against enormous shifts.
    if k >= total.bit_length() + 1:
        return 0
    return (total + (1 << (k - 1))) >> k


def isqrt_fixed(value: int) -> int:
    if value < 0:
        raise ValueError("isqrt_fixed requires a non-negative value")
    return math.isqrt(value)


def ascending_sum(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total += value
    return total


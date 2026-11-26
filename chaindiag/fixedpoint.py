from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


ALLOWED_FRAC_BITS = (8, 16, 24)
ALLOWED_QUANT_TIERS = ("q0", "q1", "q2", "q3")
ALLOWED_INT_BITS = (8, 16, 24, 32)

# Quantization-grid discard bits.  The values are deliberately different enough
# that changing tier changes retained integers, not only labels.
TIER_DISCARD_BITS = {"q0": 0, "q1": 1, "q2": 2, "q3": 4}

INT_BITS_TO_SCHEME = {8: "qint8", 16: "qint16", 24: "qint24", 32: "qint32"}
SCHEME_TO_INT_BITS = {value: key for key, value in INT_BITS_TO_SCHEME.items()}


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def stable_int(seed: int, *parts: int) -> int:
    value = splitmix64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    for part in parts:
        value = splitmix64(value ^ (int(part) & 0xFFFFFFFFFFFFFFFF))
    return value


def sign(value: int) -> int:
    return (value > 0) - (value < 0)


def round_rational(numerator: int, denominator: int, tie_rule: str = "half_away_from_zero") -> tuple[int, int, bool]:
    """Round ``numerator / denominator`` to an integer.

    Returns ``(rounded, direction, exact_half_tie)``.  Direction is the sign of
    the unrounded remainder after choosing the nearest integer: +1 means the
    selected integer is below the exact value, -1 means it is above it.
    """

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator == 0:
        return 0, 0, False

    numerator_sign = -1 if numerator < 0 else 1
    magnitude, remainder = divmod(abs(int(numerator)), denominator)
    twice = remainder * 2
    exact_half = twice == denominator
    if twice > denominator or (exact_half and tie_rule == "half_away_from_zero"):
        magnitude += 1
    elif exact_half and tie_rule == "half_even":
        if magnitude % 2:
            magnitude += 1

    rounded = numerator_sign * magnitude
    residual_num = numerator - rounded * denominator
    direction = sign(residual_num)
    return rounded, direction, exact_half


def apply_ratio(value: int, numerator: int, denominator: int, tie_rule: str) -> tuple[int, int, bool]:
    return round_rational(value * numerator, denominator, tie_rule)


def quantize_grid(value: int, discard_bits: int, tie_rule: str) -> tuple[int, int, bool]:
    if discard_bits <= 0:
        return int(value), 0, False
    grid = 1 << discard_bits
    quotient, direction, tie = round_rational(value, grid, tie_rule)
    return quotient * grid, direction, tie


def saturation_limit(int_bits: int) -> tuple[int, int]:
    upper = (1 << (int_bits - 1)) - 1
    lower = -(1 << (int_bits - 1))
    return lower, upper


def saturate(value: int, int_bits: int) -> tuple[int, bool]:
    lower, upper = saturation_limit(int_bits)
    if value < lower:
        return lower, True
    if value > upper:
        return upper, True
    return int(value), False


@dataclass
class FixedContext:
    frac_bits: int = 8
    quant_tier: str = "q2"
    tie_rule: str = "half_away_from_zero"

    def __post_init__(self) -> None:
        if not isinstance(self.frac_bits, int) or isinstance(self.frac_bits, bool) or not 1 <= self.frac_bits <= 64:
            raise ValueError("frac_bits must be an integer in [1, 64]")
        if self.quant_tier not in ALLOWED_QUANT_TIERS:
            raise ValueError(f"unsupported quant_tier: {self.quant_tier}")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def discard_bits(self) -> int:
        return TIER_DISCARD_BITS[self.quant_tier]

    def quantize(self, values: list[int], int_bits: int = 24, phase: str = "quantize") -> tuple[list[int], list[dict[str, Any]], int, list[int]]:
        output: list[int] = []
        steps: list[dict[str, Any]] = []
        ties: list[int] = []
        residual = 0
        for dim, value in enumerate(values):
            gridded, direction, tie = quantize_grid(value, self.discard_bits, self.tie_rule)
            clipped, saturated = saturate(gridded, int_bits)
            output.append(clipped)
            residual += abs(value - clipped)
            if tie:
                ties.append(dim)
            if direction != 0 or tie or saturated:
                steps.append(
                    {
                        "phase": phase,
                        "dim": dim,
                        "direction": direction,
                        "tie": tie,
                        "saturated": saturated,
                        "before": int(value),
                        "after": clipped,
                    }
                )
        return output, steps, residual, ties


def fraction_to_float_text(value: Fraction) -> dict[str, Any]:
    return {"exact": f"{value.numerator}/{value.denominator}", "float": float(value)}


def finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False

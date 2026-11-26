"""Fixed-point arithmetic with explicit truncation and saturation rules.

Internal token balances are integer multiples of 2**-scale_exponent tokens.
Replenishment is computed as floor(rate * 2**e * elapsed_ms / 1000); thus a
single refill loses strictly less than one LSB. Addition and conversion are
saturated to signed INT64_MAX because that is the declared storage format.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np


INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)


def saturate_int64(value: int) -> int:
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return int(value)


def saturating_mul(a: int, b: int) -> int:
    return saturate_int64(int(a) * int(b))


def saturating_add(a: int, b: int) -> int:
    return saturate_int64(int(a) + int(b))


def scale_factor(exponent: int) -> int:
    return 1 << exponent


def quantize_tokens(tokens: int, exponent: int) -> int:
    return saturating_mul(tokens, scale_factor(exponent))


def refill_delta_fp(rate_per_sec: int, elapsed_ms: int, exponent: int) -> int:
    """Return the floored positive replenishment in fixed-point LSBs."""
    if elapsed_ms <= 0 or rate_per_sec == 0:
        return 0
    exact_numerator = int(rate_per_sec) * scale_factor(exponent) * int(elapsed_ms)
    return saturate_int64(exact_numerator // 1000)


def refill_residual_numerator(rate_per_sec: int, elapsed_ms: int, exponent: int) -> int:
    """Remainder of the refill division (0..999), in 1/1000 LSB units."""
    if elapsed_ms <= 0 or rate_per_sec == 0:
        return 0
    exact_numerator = int(rate_per_sec) * scale_factor(exponent) * int(elapsed_ms)
    return int(exact_numerator % 1000)


@dataclass(frozen=True)
class RefillResult:
    delta_fp: int
    residual_numerator: int
    saturated: bool


def refill(balance_fp: int, capacity_fp: int, rate_per_sec: int, elapsed_ms: int, exponent: int) -> RefillResult:
    delta = refill_delta_fp(rate_per_sec, elapsed_ms, exponent)
    residual = refill_residual_numerator(rate_per_sec, elapsed_ms, exponent)
    candidate = int(balance_fp) + delta
    saturated = candidate > INT64_MAX
    candidate = saturate_int64(candidate)
    if capacity_fp >= 0 and candidate > capacity_fp:
        candidate = capacity_fp
    return RefillResult(candidate, residual, saturated)


def advance_many(
    balances: Sequence[int],
    capacities: Sequence[int],
    rates: Sequence[int],
    elapsed_ms: Sequence[int],
    exponent: int,
) -> tuple[list[int], list[int], list[bool]]:
    """Vectorized exact refill for the window path.

    ``dtype=object`` keeps Python's unbounded integer precision while NumPy
    supplies the vectorized arithmetic and comparison. Values are then
    explicitly saturated/clipped, matching the scalar rule exactly.
    """
    if not (len(balances) == len(capacities) == len(rates) == len(elapsed_ms)):
        raise ValueError("fixed-point vector length mismatch")
    if not balances:
        return [], [], []
    bal = np.asarray(list(balances), dtype=object)
    cap = np.asarray(list(capacities), dtype=object)
    rate = np.asarray(list(rates), dtype=object)
    elapsed = np.asarray(list(elapsed_ms), dtype=object)
    raw = rate * np.asarray(scale_factor(exponent), dtype=object) * elapsed
    deltas = raw // 1000
    residues = raw % 1000
    candidates = bal + deltas
    saturated_mask = candidates > INT64_MAX
    saturated = np.minimum(candidates, np.asarray(INT64_MAX, dtype=object))
    capped = np.minimum(saturated, cap)
    return (
        [int(item) for item in capped.tolist()],
        [int(item) for item in residues.tolist()],
        [bool(item) for item in saturated_mask.tolist()],
    )


def single_refill_error_bound(exponent: int) -> Fraction:
    """Worst-case token error for one refill, in token units."""
    return Fraction(999, 1000 * scale_factor(exponent))


def lsb_fraction_to_tokens(lsb: int, exponent: int) -> Fraction:
    return Fraction(int(lsb), scale_factor(exponent))


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def decimal_text(value: Fraction, places: int = 12) -> str:
    if value == 0:
        return "0"
    scale = 10**places
    rounded = (value.numerator * scale + value.denominator // 2) // value.denominator
    whole, frac = divmod(rounded, scale)
    body = f"{whole}.{frac:0{places}d}".rstrip("0").rstrip(".")
    return body


def sum_fractions(values: Iterable[Fraction]) -> Fraction:
    total = Fraction(0, 1)
    for value in values:
        total += value
    return total


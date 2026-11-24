"""Deterministic fixed point integer arithmetic.

All feature computations are performed with Python integers scaled by
``10 ** frac_bits``.  Floating point is never used inside the pipeline so that
results are identical on every platform and independent of dictionary/hash
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def quantize(value: int, gear: int) -> Tuple[int, int]:
    """Snap ``value`` to a multiple of ``gear``.

    Returns the quantized value and the signed residual (``value - quantized``).
    ``gear`` must be a positive integer.
    """

    if gear <= 0:
        raise ValueError("quantization gear must be a positive integer")
    q, r = divmod(value, gear)
    if r * 2 >= gear:
        q += 1
    quantized = q * gear
    return quantized, value - quantized


def scaled(value, frac_bits: int) -> int:
    """Convert a JSON float/int into a fixed point integer exactly."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("only numeric values can be scaled")
    scale = 10 ** frac_bits
    # Repr of a finite float round trips exactly; avoid float multiplication.
    sign = "-" if value < 0 else ""
    text = repr(abs(float(value)))
    if "e" in text or "E" in text:
        text = format(float(text), f".{frac_bits + 6}f")
    if "." in text:
        whole, frac = text.split(".")
    else:
        whole, frac = text, ""
    padded = (frac + "0" * frac_bits)[: frac_bits + 1]
    kept = padded[:frac_bits]
    round_up = bool(padded[frac_bits:]) and int(padded[frac_bits]) >= 5
    magnitude = int(whole or "0") * (10 ** frac_bits) + int(kept or "0") + (1 if round_up else 0)
    return -magnitude if sign == "-" else magnitude


def to_float(value: int, frac_bits: int) -> float:
    """Inverse of :func:`scaled`, used only for human readable reports."""

    return value / (10 ** frac_bits)


def divide_round(num: int, den: int) -> int:
    """Signed integer division rounded to nearest, ties away from zero."""

    if den == 0:
        raise ZeroDivisionError("fixed point division by zero")
    q, r = divmod(abs(num), abs(den))
    if r * 2 >= abs(den):
        q += 1
    return q if (num >= 0) == (den > 0) else -q


def sqrt_fixed(value: int) -> int:
    """Integer square root for a non-negative fixed point value."""

    if value < 0:
        raise ValueError("sqrt of negative fixed point value")
    lo, hi = 0, 1
    while hi * hi <= value:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid <= value:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class FixedConfig:
    """Global fixed point configuration."""

    frac_bits: int
    gear: int

    def validate(self) -> None:
        if not isinstance(self.frac_bits, int) or self.frac_bits < 1 or self.frac_bits > 12:
            raise ValueError("frac_bits must be an integer in [1, 12]")
        if not isinstance(self.gear, int) or self.gear < 1:
            raise ValueError("gear must be a positive integer")

    @property
    def scale(self) -> int:
        return 10 ** self.frac_bits

    def quantize(self, value: int) -> Tuple[int, int]:
        return quantize(value, self.gear)

    def step(self) -> int:
        """Quantization step expressed in fixed point units (always gear)."""

        return self.gear

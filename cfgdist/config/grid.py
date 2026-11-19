"""Precision grid: the single source of truth for fixed-point scaling.

The grid is defined by an integer exponent ``q``; the scale factor is
``S = 10 ** q`` (only powers of ten are accepted).  All real values and all
derivation coefficients are quantized against the same grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..errors import E_GRID, ErrorDetail, PlatformError

# Practical bound: keeps integer arithmetic comfortably inside Python big ints
# while still allowing very fine grids.
_MIN_Q = 0
_MAX_Q = 12


@dataclass(frozen=True)
class Grid:
    q: int
    scale: int

    @classmethod
    def from_q(cls, q: int) -> "Grid":
        details: list[ErrorDetail] = []
        if isinstance(q, bool) or not isinstance(q, int):
            details.append(
                ErrorDetail(
                    reason="grid exponent must be an integer power-of-ten exponent",
                    source="grid",
                )
            )
        elif not _MIN_Q <= q <= _MAX_Q:
            details.append(
                ErrorDetail(
                    reason=f"grid exponent out of range [{_MIN_Q},{_MAX_Q}]",
                    source="grid",
                )
            )
        PlatformError.raise_if(
            E_GRID, "illegal precision grid parameter", details
        )
        return cls(q=q, scale=10**q)

    def to_dict(self) -> dict[str, int]:
        return {"q": self.q, "scale": self.scale}

    def quantize_int(self, value: Fraction) -> int:
        """Quantize an exact rational to the grid, truncating toward zero.

        Truncation direction is explicitly *toward zero* (C ``trunc``), which
        keeps the absolute quantization error bounded by one ULP.
        """
        scaled = value * self.scale
        numerator, denominator = scaled.numerator, scaled.denominator
        # Explicit truncation toward zero (C trunc), independent of sign.
        quotient = abs(numerator) // denominator
        return -quotient if numerator < 0 else quotient

    def quantize(self, number: int | float | str | Fraction) -> int:
        return self.quantize_int(_to_fraction(number))


def _to_fraction(number: int | float | str | Fraction) -> Fraction:
    if isinstance(number, Fraction):
        return number
    if isinstance(number, bool):
        raise ValueError("boolean is not a supported numeric type")
    if isinstance(number, int):
        return Fraction(number)
    if isinstance(number, float):
        # Route through repr() so the float is read at its shortest exact
        # decimal representation rather than its binary expansion.
        return Fraction(repr(number))
    if isinstance(number, str):
        return Fraction(number)
    raise ValueError(f"unsupported numeric type: {type(number)!r}")

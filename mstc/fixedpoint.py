"""Deterministic fixed-point integer arithmetic.

All classification math is performed with signed 64-bit integers.  Two knobs
control the numeric representation:

* ``fractional_bits``  - the number of bits used for the fractional part;
* ``quantization_gear`` - an enum that adds an extra right-shift (coarser
  integer grid).  Switching gear therefore changes the real integers stored in
  every intermediate and the final class scores, while staying fully
  deterministic.

Each quantized value remembers its residual (the dropped low bits, encoded in a
fixed ``MICRO`` scale so it is itself an integer) and the rounding direction
(``-1`` floor / ``+1`` round-up / ``0`` exact).  No float is ever used on the
hot path; a float64 reference implementation lives in :mod:`mstc.diagnostics`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)

# Residuals are reported in micro-units (1e-6) to keep records integer.
MICRO = 1_000_000


class Gear(str, Enum):
    FINE = "FINE"
    NORMAL = "NORMAL"
    COARSE = "COARSE"


# Extra right-shift applied on top of the fractional bit width.
GEAR_SHIFT = {Gear.FINE: 0, Gear.NORMAL: 1, Gear.COARSE: 2}
ALLOWED_GEARS = tuple(g.value for g in Gear)
ALLOWED_FRAC_BITS = (8, 12, 16)


class Rounding:
    FLOOR = -1
    EXACT = 0
    UP = +1


def clamp_i64(value: int) -> int:
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return value


def isqrt(n: int) -> int:
    """Non-negative integer square root (Newton, deterministic)."""
    if n < 0:
        raise ValueError("isqrt of negative number")
    if n == 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


@dataclass(frozen=True)
class FixedConfig:
    fractional_bits: int
    gear: Gear

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def gear_shift(self) -> int:
        return GEAR_SHIFT[self.gear]

    @property
    def total_shift(self) -> int:
        return self.fractional_bits + self.gear_shift

    @property
    def half_step(self) -> int:
        """Half of one integer LSB, expressed in fixed-point units."""
        return 1 << (self.total_shift - 1) if self.total_shift > 0 else 0

    def residual_micro(self, residual: int) -> int:
        """Convert a raw residual (in pre-shift units) to micro-units."""
        if self.total_shift == 0:
            return 0
        # residual / 2^total_shift, in micro units
        return (residual * MICRO) >> self.total_shift


def from_int(value: int, cfg: FixedConfig) -> Tuple[int, int, int]:
    """Quantize an already-integer raw value into representation scale."""
    raw = int(value) << cfg.fractional_bits
    shift = cfg.gear_shift
    if shift == 0:
        return clamp_i64(raw), 0, Rounding.EXACT
    half = 1 << (shift - 1)
    q, rem = divmod(raw, 1 << shift)
    if rem == 0:
        direction = Rounding.EXACT
    elif rem >= half:
        q += 1
        direction = Rounding.UP
    else:
        direction = Rounding.FLOOR
    return clamp_i64(q), rem, direction


def to_int(q: int, cfg: FixedConfig) -> int:
    """Inverse of :func:`from_int` (best-effort integer reconstruction)."""
    return q >> cfg.gear_shift


def q_div(numer: int, denom: int, cfg: FixedConfig, shift_extra: int = 0) -> Tuple[int, int, int]:
    """Fixed point division ``numer/denom`` scaled by ``2^fractional_bits``.

    ``numer`` and ``denom`` are raw integers (e.g. band values).  The result is
    returned in the representation scale (gear applied).  Deterministic
    round-half-up is used.
    """
    if denom == 0:
        return 0, 0, Rounding.EXACT
    total = cfg.fractional_bits + shift_extra
    num = numer << total
    rounded_num = num + (denom // 2 if denom > 0 else -((-denom) // 2))
    q, rem = divmod(rounded_num, denom)
    shift = cfg.gear_shift
    if shift:
        half = 1 << (shift - 1)
        q2, rem2 = divmod(q, 1 << shift)
        if rem2 and rem2 >= half:
            q2 += 1
            direction = Rounding.UP
        elif rem2:
            direction = Rounding.FLOOR
        else:
            direction = Rounding.EXACT
        return clamp_i64(q2), rem2, direction
    if rem:
        direction = Rounding.UP if (rem * 2) >= abs(denom) else Rounding.FLOOR
    else:
        direction = Rounding.EXACT
    return clamp_i64(q), rem, direction


def q_mul(a: int, b: int, cfg: FixedConfig) -> Tuple[int, int, int]:
    """Multiply two representation-scale fixed values, re-quantize to scale."""
    prod = a * b
    shift = cfg.fractional_bits + cfg.gear_shift
    if shift == 0:
        return clamp_i64(prod), 0, Rounding.EXACT
    half = 1 << (shift - 1)
    q, rem = divmod(prod, 1 << shift)
    if rem == 0:
        direction = Rounding.EXACT
    elif rem >= half:
        q += 1
        direction = Rounding.UP
    else:
        direction = Rounding.FLOOR
    return clamp_i64(q), rem, direction


def q_weighted(value: int, weight_q: int, cfg: FixedConfig) -> Tuple[int, int, int]:
    """Multiply a fixed value by a fixed weight and re-quantize."""
    return q_mul(value, weight_q, cfg)


def weight_to_fixed(weight: float, cfg: FixedConfig) -> int:
    """Convert a non-negative float weight into a fixed-point integer."""
    scaled = weight * cfg.scale
    # deterministic round half up
    rounded = int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)
    shifted = rounded >> cfg.gear_shift
    return clamp_i64(shifted)


def argmax_tie_low(scores: list[int]) -> int:
    """Index of the maximum score; ties resolved by lowest index (class code)."""
    best_idx = 0
    best_val = scores[0]
    for idx in range(1, len(scores)):
        if scores[idx] > best_val:
            best_val = scores[idx]
            best_idx = idx
    return best_idx

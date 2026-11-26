"""Deterministic fixed-point integer arithmetic.

A fixed point value is a plain Python ``int`` holding the *raw* integer
representation ``raw = real * 2**frac_bits``.  No float is ever used in the
pipeline arithmetic; floats only appear in the optional high-width reference
path described in :mod:`vdu.error_track`.

Two quantization tiers are provided.  The fine tier is intentionally one bit
wider than the coarse tier and keeps more fractional bits, while the
reference tier uses even more fractional bits and disables intermediate
requantization, which is the "one tier wider, no quantization" reference
required by the error tracking stage.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantTier:
    """Static description of a fixed-point quantization tier."""

    name: str
    frac_bits: int
    total_bits: int

    @property
    def scale(self):
        return 1 << self.frac_bits

    @property
    def half_step(self):
        return 1 << max(0, self.frac_bits - 1)

    def to_raw(self, real):
        """Quantize a real number into the raw integer of this tier."""
        return int(round(real * self.scale))

    def to_real(self, raw):
        return raw / self.scale


COARSE = QuantTier(name="coarse", frac_bits=8, total_bits=32)
FINE = QuantTier(name="fine", frac_bits=10, total_bits=33)
REFERENCE = QuantTier(name="reference", frac_bits=14, total_bits=40)

TIERS = {COARSE.name: COARSE, FINE.name: FINE, REFERENCE.name: REFERENCE}


def get_tier(name):
    try:
        return TIERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown quantization tier {name!r}") from exc


def from_int(value, tier):
    """Promote a plain integer count into fixed point raw representation."""
    return int(value) * tier.scale


def add(a, b):
    """Fixed point addition; operands must share the same fractional width."""
    return int(a) + int(b)


def sub(a, b):
    return int(a) - int(b)


def mul(a, b, tier):
    """Multiply two fixed point raw values, returning same-tier raw.

    The doubled fractional shift is removed with deterministic round-half-up
    so every multiplication stays in the integer domain.
    """
    doubled = int(a) * int(b)
    shift = tier.frac_bits
    if shift == 0:
        return doubled
    rounded = (doubled + (1 << (shift - 1))) >> shift
    return rounded


def mul_by_int(raw, integer):
    """Scale a fixed point raw value by a plain integer (no rounding)."""
    return int(raw) * int(integer)


def div_by_int(raw, integer, tier):
    """Divide a fixed point raw value by a plain integer with rounding."""
    if integer == 0:
        raise ZeroDivisionError("fixed point division by zero")
    if integer < 0:
        return -div_by_int(-raw, -integer, tier)
    return (int(raw) + integer // 2) // integer


def requantize(raw, src, dst):
    """Convert raw integer between fractional widths with residual tracking.

    Returns ``(new_raw, residual)`` where residual is expressed in destination
    fractional units so callers can persist the exact quantization remainder.
    """
    if src.frac_bits == dst.frac_bits:
        return int(raw), 0
    if src.frac_bits > dst.frac_bits:
        drop = src.frac_bits - dst.frac_bits
        factor = 1 << drop
        new_raw = (int(raw) + (1 << (drop - 1))) >> drop
        residual = int(raw) - (new_raw << drop)
        return new_raw, residual
    shift = dst.frac_bits - src.frac_bits
    return int(raw) << shift, 0


def quant_step(tier):
    """The numeric quantization step as a fixed point raw value (always 1)."""
    return 1


def clamp_to_width(raw, tier):
    """Saturate a raw integer to the signed ``total_bits`` range."""
    bound = 1 << (tier.total_bits - 1)
    if raw >= bound:
        return bound - 1
    if raw < -bound:
        return -bound
    return raw


def signed_distance(a, b):
    """Absolute integer distance between two raw values."""
    return abs(int(a) - int(b))

"""Fixed-point integer arithmetic and quantization gears.

All monetary-like routing math is performed on plain Python integers so that
results are bit-for-bit identical across platforms, interpreters and threads.
A fixed-point number with ``fraction_bits`` fractional bits represents the
rational value ``raw / 2**fraction_bits`` where ``raw`` is the stored integer.

Quantization gears reduce the effective resolution.  Gear ``g`` (1, 2 or 4)
rounds the raw integer to a multiple of ``g`` using deterministic round-half-up
(half is always rounded toward positive infinity).  The reference ("high
bit-width") score keeps full resolution, while the authoritative score is the
quantized fixed-point integer.
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_FRACTION_BITS = (4, 8, 16)
ALLOWED_GEARS = (1, 2, 4)


def round_half_up(value: int, step: int) -> int:
    """Round ``value`` to a multiple of positive ``step`` (half rounds up)."""
    if step <= 0:
        raise ValueError("quantization step must be positive")
    if step == 1:
        return value
    q, r = divmod(value, step)
    if r * 2 >= step:
        q += 1
    return q * step


@dataclass(frozen=True)
class QuantGear:
    """A fixed-point domain: fractional bit width plus a quantization gear."""

    fraction_bits: int
    gear: int

    def __post_init__(self) -> None:
        if self.fraction_bits not in ALLOWED_FRACTION_BITS:
            raise ValueError(
                f"fraction_bits must be one of {ALLOWED_FRACTION_BITS}, "
                f"got {self.fraction_bits}"
            )
        if self.gear not in ALLOWED_GEARS:
            raise ValueError(
                f"gear must be one of {ALLOWED_GEARS}, got {self.gear}"
            )

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    @property
    def step(self) -> int:
        """Quantization step in raw fixed-point units."""
        return self.gear

    @property
    def half_step(self) -> int:
        return (self.gear + 1) // 2 if self.gear > 1 else 0

    def from_int(self, value: int) -> int:
        """Lift an integer quantity into raw fixed-point units."""
        return int(value) * self.scale

    def quantize(self, raw: int) -> int:
        """Quantize a full-resolution raw value with round-half-up."""
        return round_half_up(raw, self.gear)

    def at_half_step(self, raw: int) -> bool:
        """True when ``raw`` sits exactly on a quantization half-step boundary."""
        if self.gear <= 1:
            return False
        return raw % self.gear == (self.gear // 2)

    def to_float(self, raw: int) -> float:
        """Lossy decimal view for reporting only; never used for decisions."""
        return raw / self.scale

    @property
    def key(self) -> str:
        return f"fb{self.fraction_bits}-g{self.gear}"


@dataclass(frozen=True)
class FixedPoint:
    """Convenience wrapper keeping a reference raw value and quantized value."""

    gear: QuantGear
    reference_raw: int
    quantized_raw: int

    @classmethod
    def of(cls, gear: QuantGear, reference_raw: int) -> "FixedPoint":
        return cls(gear, reference_raw, gear.quantize(reference_raw))

    @property
    def quant_error(self) -> int:
        """Signed difference introduced purely by quantization (ref - quant)."""
        return self.reference_raw - self.quantized_raw

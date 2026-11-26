"""Deterministic fixed-point integer arithmetic.

All numeric kernels in the pipeline operate on signed integers with an
explicit fractional bit width. Switching the bit width or quantization tier
must change the actual integer values, which is verified by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .codes import BizCode


class FpWidth(int, Enum):
    W16 = 16
    W32 = 32
    W64 = 64


class QuantTier(str, Enum):
    COARSE = "coarse"
    STANDARD = "standard"
    FINE = "fine"


# Fractional bits assigned per (width, tier). Each entry is a distinct,
# enumerated operating point; unknown combinations are rejected, not clamped.
_FRAC_BITS: dict[tuple[int, str], int] = {
    (16, "coarse"): 6,
    (16, "standard"): 8,
    (16, "fine"): 10,
    (32, "coarse"): 10,
    (32, "standard"): 14,
    (32, "fine"): 18,
    (64, "coarse"): 20,
    (64, "standard"): 28,
    (64, "fine"): 36,
}

ALLOWED_WIDTHS = tuple(w.value for w in FpWidth)
ALLOWED_TIERS = tuple(t.value for t in QuantTier)


def validate_fp_enum(width: int, tier: str) -> None:
    if width not in ALLOWED_WIDTHS:
        raise ValueError(f"{BizCode.E_FP_ENUM.value}: width={width!r}")
    if tier not in ALLOWED_TIERS:
        raise ValueError(f"{BizCode.E_FP_ENUM.value}: tier={tier!r}")
    if (width, tier) not in _FRAC_BITS:  # pragma: no cover - defensive
        raise ValueError(f"{BizCode.E_FP_ENUM.value}: pair=({width},{tier})")


def frac_bits_for(width: int, tier: str) -> int:
    validate_fp_enum(width, tier)
    return _FRAC_BITS[(width, tier)]


@dataclass(frozen=True)
class FpSpec:
    width: int = 32
    tier: str = "standard"

    def __post_init__(self) -> None:
        validate_fp_enum(self.width, self.tier)

    @property
    def frac_bits(self) -> int:
        return _FRAC_BITS[(self.width, self.tier)]

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def int_min(self) -> int:
        return -(1 << (self.width - 1))

    @property
    def int_max(self) -> int:
        return (1 << (self.width - 1)) - 1

    def as_dict(self) -> dict:
        return {"width": self.width, "tier": self.tier,
                "frac_bits": self.frac_bits}

    def key_material(self) -> str:
        return f"fp:{self.width}:{self.tier}:q{self.frac_bits}"


def clamp_int(value: int, spec: FpSpec) -> int:
    if value > spec.int_max:
        return spec.int_max
    if value < spec.int_min:
        return spec.int_min
    return value


def quantize(value: float, spec: FpSpec) -> int:
    """Round-half-away-from-zero to a fixed-point integer, then clamp."""
    scaled = value * spec.scale
    if scaled >= 0:
        q = int(scaled + 0.5)
    else:
        q = int(scaled - 0.5)
    return clamp_int(q, spec)


def dequantize(value: int, spec: FpSpec) -> float:
    return value / spec.scale


def quantize_many(values, spec: FpSpec) -> list[int]:
    return [quantize(float(v), spec) for v in values]


def rescale(value: int, src: FpSpec, dst: FpSpec) -> int:
    """Move a fixed-point integer between operating points without floats.

    The quantization residual of the down/up shift is returned alongside the
    new integer so callers can log every lossy step.
    """
    if src.frac_bits == dst.frac_bits:
        return clamp_int(value, dst), 0
    if dst.frac_bits > src.frac_bits:
        shift = dst.frac_bits - src.frac_bits
        return clamp_int(value << shift, dst), 0
    shift = src.frac_bits - dst.frac_bits
    factor = 1 << shift
    if value >= 0:
        new_value = (value + (factor >> 1)) // factor
        residual = value - (new_value << shift)
    else:
        magnitude = -value
        rounded = (magnitude + (factor >> 1)) // factor
        new_value = -rounded
        residual = value + (rounded << shift)
    return clamp_int(new_value, dst), residual


def fixed_mul(a: int, b: int, spec: FpSpec) -> tuple[int, int]:
    """Multiply two fixed-point integers, returning (product, residual).

    The raw product carries 2*frac_bits and is shifted back. The dropped
    low bits are reported as the quantization residual.
    """
    raw = a * b
    if spec.frac_bits == 0:
        return clamp_int(raw, spec), 0
    factor = 1 << spec.frac_bits
    if raw >= 0:
        result = (raw + (factor >> 1)) // factor
    else:
        result = -((-raw + (factor >> 1)) // factor)
    residual = raw - (result << spec.frac_bits)
    return clamp_int(result, spec), residual


def fixed_add(a: int, b: int, spec: FpSpec) -> int:
    return clamp_int(a + b, spec)

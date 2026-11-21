"""Fixed-point integer quantization and significance judgement.

Every drift score produced by the metric layer is converted to a fixed-point
integer before significance is decided.  A quantization tier defines the
fractional decimal width and an integer clamp range, so switching tiers changes
the stored integer value (not only its display formatting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantTier:
    name: str
    decimals: int
    clamp: int

    @property
    def scale(self) -> int:
        return 10 ** self.decimals


# Ordered coarse -> fine.  The implicit reference tier is one width above the
# finest configured tier (see reference_tier()).
DEFAULT_TIERS: tuple[QuantTier, ...] = (
    QuantTier("coarse", decimals=2, clamp=10_000),
    QuantTier("standard", decimals=4, clamp=1_000_000),
    QuantTier("fine", decimals=6, clamp=100_000_000),
)


def reference_tier(tiers: tuple[QuantTier, ...]) -> QuantTier:
    top = tiers[-1]
    return QuantTier(
        name=f"{top.name}_ref",
        decimals=top.decimals + 2,
        clamp=top.clamp * 100,
    )


def quantize(value: float, tier: QuantTier) -> int:
    """Round half away from zero into fixed-point integers, then clamp."""
    scaled = value * tier.scale
    if scaled >= 0:
        q = math.floor(scaled + 0.5)
    else:
        q = math.ceil(scaled - 0.5)
    if q > tier.clamp:
        return tier.clamp
    if q < -tier.clamp:
        return -tier.clamp
    return q


def retained(q: int, tier: QuantTier) -> float:
    """Exact retained decimal value for a quantized integer."""
    return q / tier.scale


def quantize_threshold(threshold: float, tier: QuantTier) -> int:
    return quantize(threshold, tier)


def is_significant(score_q: int, threshold_q: int) -> bool:
    return abs(score_q) >= threshold_q


def rescale(q: int, src: QuantTier, dst: QuantTier) -> int:
    """Map an integer from one tier width onto another (truncating)."""
    if dst.decimals >= src.decimals:
        return q * (10 ** (dst.decimals - src.decimals))
    factor = 10 ** (src.decimals - dst.decimals)
    if q >= 0:
        return q // factor
    return -((-q) // factor)

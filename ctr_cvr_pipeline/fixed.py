from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


TIER_BITS: dict[str, int] = {"Q8": 8, "Q12": 12, "Q16": 16, "Q24": 24}
ALLOWED_TIERS: tuple[str, ...] = tuple(TIER_BITS)
ALLOWED_FRACTIONAL_BITS: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16)


@dataclass(frozen=True)
class Rounding:
    value: int
    residual: int
    direction: str
    clipped: bool = False


def fraction_scale(fractional_bits: int) -> int:
    return 1 << fractional_bits


def tier_qmax(tier: str) -> int:
    return (1 << (TIER_BITS[tier] - 1)) - 1


def tier_qmin(tier: str) -> int:
    return -(1 << (TIER_BITS[tier] - 1))


def clamp_fixed(value: int, tier: str) -> tuple[int, bool, str | None]:
    qmax = tier_qmax(tier)
    qmin = tier_qmin(tier)
    if value > qmax:
        return qmax, True, "CLIPPED_HIGH"
    if value < qmin:
        return qmin, True, "CLIPPED_LOW"
    return value, False, None


def round_div_half_away(numerator: int, denominator: int) -> Rounding:
    """Round numerator/denominator to nearest integer, halves away from zero."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    rounded_mag = (2 * magnitude + denominator) // (2 * denominator)
    value = sign * rounded_mag
    residual = numerator - value * denominator
    if residual > 0:
        direction = "DOWN"
    elif residual < 0:
        direction = "UP"
    else:
        direction = "EXACT"
    return Rounding(value=value, residual=residual, direction=direction)


def quantize_real(value: float, fractional_bits: int, tier: str) -> Rounding:
    """Quantize a finite real into signed Qm.n with explicit range clipping."""
    if not math.isfinite(value):
        raise ValueError("cannot quantize a non-finite value")
    scale = fraction_scale(fractional_bits)
    scaled = value * scale
    rounded = int(math.floor(scaled + 0.5)) if scaled >= 0 else int(math.ceil(scaled - 0.5))
    clipped_value, clipped, clip_direction = clamp_fixed(rounded, tier)
    if clipped:
        dequantized = clipped_value / scale
        residual_float = value - dequantized
        direction = clip_direction or "EXACT"
    else:
        dequantized = clipped_value / scale
        residual_float = value - dequantized
        direction = "UP" if residual_float < 0 else ("DOWN" if residual_float > 0 else "EXACT")
    return Rounding(
        value=clipped_value,
        residual=int(round(residual_float * scale)),
        direction=direction,
        clipped=clipped,
    )


def dequantize(value: int, fractional_bits: int) -> float:
    return value / fraction_scale(fractional_bits)


def signed_add_trace(a: int, b: int, tier: str) -> tuple[int, bool, str | None]:
    return clamp_fixed(a + b, tier)


@lru_cache(maxsize=None)
def sigmoid_lut(fractional_bits: int, tier: str) -> tuple[tuple[int, ...], int, int]:
    """Nearest-neighbour logistic lookup over [-8, 8] at 1/16 resolution.

    The LUT is materialized once from a deterministic cutoff grid. Every model
    forward pass after this point is integer-only LUT indexing.
    """
    scale = fraction_scale(fractional_bits)
    resolution = 16
    if scale % resolution:
        raise ValueError("fractional bits must support 1/16 LUT resolution")
    step = scale // resolution
    values: list[int] = []
    for index in range(8 * resolution * 2 + 1):
        real = -8.0 + index / resolution
        p = 1.0 / (1.0 + math.exp(-real))
        q = int(math.floor(p * scale + 0.5))
        q, _, _ = clamp_fixed(q, tier)
        values.append(q)
    return tuple(values), step, scale


def sigmoid_fixed(preactivation: int, fractional_bits: int, tier: str) -> tuple[int, int, int]:
    """Return (score_q, lut_index, clamped_preactivation_q)."""
    lut, step, scale = sigmoid_lut(fractional_bits, tier)
    low = -8 * scale
    high = 8 * scale
    clamped = min(high, max(low, preactivation))
    shifted = clamped + 8 * scale
    index = (2 * shifted + step) // (2 * step)
    index = min(len(lut) - 1, max(0, index))
    return lut[index], index, clamped


def quantize_model_hyperparameter(value: float, fractional_bits: int, tier: str) -> int:
    return quantize_real(value, fractional_bits, tier).value


def trace_rounding(
    stage: str,
    operation: str,
    numerator: int,
    denominator: int,
    *,
    sample: str | None = None,
    parameter: str | None = None,
    epoch: int | None = None,
    batch: int | None = None,
    extra: dict | None = None,
) -> tuple[int, dict]:
    rounded = round_div_half_away(numerator, denominator)
    record: dict = {
        "stage": stage,
        "operation": operation,
        "numerator": numerator,
        "denominator": denominator,
        "rounded": rounded.value,
        "residual": rounded.residual,
        "direction": rounded.direction,
    }
    if sample is not None:
        record["sample"] = sample
    if parameter is not None:
        record["parameter"] = parameter
    if epoch is not None:
        record["epoch"] = epoch
    if batch is not None:
        record["batch"] = batch
    if extra:
        record.update(extra)
    return rounded.value, record

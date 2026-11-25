"""Deterministic fixed-point integer arithmetic.

All acoustic accumulation, projection and posterior arithmetic happens in the
*integer domain* so results are bit-for-bit reproducible across machines and
Python versions (no platform ``float`` ordering, no SIMD non-determinism).

A signed Qm.n value with ``frac_bits = n`` stores ``round(x * 2**n)``.  We use
Python's arbitrary-precision integers internally but saturate every store back
to a signed 32-bit container, mirroring an embedded fixed-point DSP and giving
a well-defined, testable overflow policy.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from .errors import VADValidationError, E_FIELD_INVALID

# Signed 32-bit storage container.
INT_MIN = -(2 ** 31)
INT_MAX = 2 ** 31 - 1

# Allowed fractional bit widths and quantization (rounding) modes, kept here so
# the config layer and this module cannot drift apart.
ALLOWED_FRAC_BITS = (8, 15, 23)
ALLOWED_QUANT_MODES = ("TRUNC", "ROUND_HALF_EVEN", "ROUND_HALF_UP")


def check_quant_params(frac_bits: int, quant_mode: str) -> None:
    if not isinstance(frac_bits, int) or isinstance(frac_bits, bool):
        raise VADValidationError(
            "frac_bits must be an integer", E_FIELD_INVALID,
            detail={"frac_bits": frac_bits})
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise VADValidationError(
            f"frac_bits {frac_bits} not in {ALLOWED_FRAC_BITS}",
            E_FIELD_INVALID, detail={"frac_bits": frac_bits})
    if quant_mode not in ALLOWED_QUANT_MODES:
        raise VADValidationError(
            f"quant_mode {quant_mode!r} not in {ALLOWED_QUANT_MODES}",
            E_FIELD_INVALID, detail={"quant_mode": quant_mode})


def saturate(value: int) -> int:
    """Clamp an arbitrary Python int into the signed 32-bit container."""
    if value > INT_MAX:
        return INT_MAX
    if value < INT_MIN:
        return INT_MIN
    return value


def to_fixed(x: float, frac_bits: int, quant_mode: str) -> int:
    """Quantize a real value to a saturated fixed-point integer."""
    scale = float(1 << frac_bits)
    scaled = x * scale
    if quant_mode == "TRUNC":
        q = int(math.trunc(scaled))
    elif quant_mode == "ROUND_HALF_UP":
        # Away from zero on the exact half.
        q = int(math.floor(scaled + 0.5)) if scaled >= 0 else \
            int(math.ceil(scaled - 0.5))
    else:  # ROUND_HALF_EVEN - banker's rounding, the deterministic default.
        q = int(round(scaled))
    return saturate(q)


def to_float(q: int, frac_bits: int) -> float:
    """De-quantize a fixed-point integer back to a real value."""
    return float(q) / float(1 << frac_bits)


def fp_add(a: int, b: int) -> int:
    """Saturating fixed-point addition (operands share the same Q format)."""
    return saturate(int(a) + int(b))


def fp_sum(values: Iterable[int]) -> int:
    """Deterministic saturating left-to-right integer sum."""
    acc = 0
    for v in values:
        acc = fp_add(acc, int(v))
    return acc


def fp_mul(a: int, b: int, frac_bits: int) -> int:
    """Fixed-point multiply: Qn * Qn -> Qn with saturation and re-scaling."""
    product = int(a) * int(b)
    # Arithmetic right shift for negative numbers (floor division by 2**n).
    scaled = product >> frac_bits
    return saturate(scaled)


def fp_div(numerator: int, denominator: int, frac_bits: int,
           quant_mode: str) -> int:
    """Fixed-point division of two Qn operands, result kept in Qn.

    Implemented in the real domain of the de-quantized ratio then re-quantized,
    which is exact for the deterministic ratios we form (counts are small).
    """
    if denominator == 0:
        return 0
    ratio = to_float(numerator, frac_bits) / to_float(denominator, frac_bits)
    return to_fixed(ratio, frac_bits, quant_mode)


def half_step(frac_bits: int) -> float:
    """Real-valued magnitude of one quantization half step."""
    return 0.5 / float(1 << frac_bits)


def vector_quantize(values: Sequence[float], frac_bits: int,
                    quant_mode: str) -> List[int]:
    return [to_fixed(v, frac_bits, quant_mode) for v in values]

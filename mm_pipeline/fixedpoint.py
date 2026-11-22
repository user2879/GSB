"""Deterministic fixed-point integer arithmetic.

Every visual/text feature and every fusion quantity is an integer scaled by
``S = 2 ** frac_bits``. No platform specific floating point behaviour can
therefore influence a conclusion: exponentials, softmax, division and rounding
are all reduced to exact integer operations with documented tie rules.
"""

from __future__ import annotations

import math

# Allowed enums (validated again in the config layer; duplicated intentionally
# so this module stays import-safe on its own).
QUANT_LEVELS = (8, 12, 16)
FRAC_BIT_WIDTHS = (8, 16, 24)


def scale(frac_bits: int) -> int:
    return 1 << frac_bits


def quant_shift(quant_level: int) -> int:
    """Map the quantization-level enum to a right-shift applied after pooling."""
    if quant_level not in QUANT_LEVELS:
        raise ValueError(f"quant_level must be one of {QUANT_LEVELS}")
    return quant_level // 4  # 8->2, 12->3, 16->4


def from_float(value: float, frac_bits: int) -> int:
    """Deterministic half-away-from-zero float -> fixed point."""
    if not math.isfinite(value):
        raise ValueError("only finite values can be encoded as fixed point")
    s = scale(frac_bits)
    scaled = value * s
    if scaled >= 0:
        return int(math.floor(scaled + 0.5))
    return -int(math.floor(-scaled + 0.5))


def to_float(value: int, frac_bits: int) -> float:
    return value / scale(frac_bits)


def fmul(a: int, b: int, frac_bits: int) -> int:
    """Multiply two fixed-point operands (both already scaled)."""
    return (a * b) >> frac_bits


def fdiv_scaled(a: int, b: int, frac_bits: int) -> int:
    """Divide two fixed-point operands: (a/S)/(b/S) = a/b, returned scaled by S.

    Use when both ``a`` and ``b`` are already fixed-point integers. Floor
    semantics are sign-correct under Python integer division.
    """
    if b == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    return (a * scale(frac_bits)) // b


def fdiv_int(numerator: int, divisor: int, frac_bits: int) -> int:
    """Divide a raw integer numerator by a raw integer divisor into fixed point.

    Returns (numerator / divisor) * S with floor semantics for signed inputs.
    """
    if divisor == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    return (numerator * scale(frac_bits)) // divisor


def fdiv_round(numerator: int, divisor: int, frac_bits: int) -> int:
    """Raw integer ratio into fixed point, half rounded away from zero."""
    if divisor == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    num = numerator * scale(frac_bits)
    neg = (num < 0) != (divisor < 0)
    num_abs, d_abs = abs(num), abs(divisor)
    q = (2 * num_abs + d_abs) // (2 * d_abs)
    return -q if neg else q


def clamp_confidence(value: int, frac_bits: int) -> int:
    return max(0, min(scale(frac_bits), value))


def _ln2_fixed(frac_bits: int) -> int:
    return from_float(math.log(2.0), frac_bits)


def exp_fp(z: int, frac_bits: int) -> int:
    """Deterministic fixed-point exp(z) using range reduction and a Taylor sum.

    Reduction: z = k*ln(2) + r with r in [-ln(2), 0]; then
    exp(z) = 2**k * exp(r), and exp(r) is a degree-10 integer Taylor sum.
    """
    ln2 = _ln2_fixed(frac_bits)
    s = scale(frac_bits)
    k = z // ln2
    r = z - k * ln2

    term = s          # term n=0 == 1.0
    total = s         # exp(r) accumulator
    for n in range(1, 11):
        # term_n = term_(n-1) * (r / n); r and term are both fixed point, n raw.
        term = (term * r) // (n * s)
        total += term

    if k >= 0:
        return total << k
    return total >> (-k)


def softmax_two(logit_a: int, logit_b: int, frac_bits: int, temp: int) -> tuple[int, int]:
    """Numerically stable (max-subtracted) two-element softmax with temperature.

    ``logit_*`` and ``temp`` are fixed point. Returns weights (w_a, w_b) that
    sum to exactly ``S``; any residual rounding unit stays deterministically
    with modality A.
    """
    za = fdiv_scaled(logit_a, temp, frac_bits)
    zb = fdiv_scaled(logit_b, temp, frac_bits)
    m = max(za, zb)
    ea = exp_fp(za - m, frac_bits)
    eb = exp_fp(zb - m, frac_bits)
    denom = ea + eb
    wa = (ea * scale(frac_bits)) // denom
    wb = scale(frac_bits) - wa
    return wa, wb

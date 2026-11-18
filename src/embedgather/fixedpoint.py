"""Deterministic fixed-point arithmetic.

A real value x is represented by the integer q = round_half_away(x * 2**F)
clamped to the representable range implied by the quantization level:
levels = 2**bits, q in [-(levels//2), levels//2 - 1].

All rounding uses round-half-away-from-zero so results are identical on
every platform and across runs.
"""

import math


def is_power_of_two(value):
    return isinstance(value, int) and value >= 2 and (value & (value - 1)) == 0


def round_half_away(value):
    if value >= 0:
        return math.floor(value + 0.5)
    return -math.floor(-value + 0.5)


def quantize_scalar(x, frac_bits, levels):
    scaled = x * (1 << frac_bits)
    q = round_half_away(scaled)
    lo = -(levels // 2)
    hi = levels // 2 - 1
    if q < lo:
        return lo
    if q > hi:
        return hi
    return q


def quantize_vector(vec, frac_bits, levels):
    return tuple(quantize_scalar(v, frac_bits, levels) for v in vec)


def dequantize_scalar(q, frac_bits):
    return q / float(1 << frac_bits)


def fixed_mean(total, count):
    """Fixed-point division by count with round-half-away-from-zero."""
    if count <= 0:
        raise ValueError("fixed_mean requires count > 0")
    q, r = divmod(total, count)
    twice = 2 * r
    if twice > count or (twice == count and total >= 0):
        q += 1
    return q


def ulp(frac_bits):
    return 1.0 / float(1 << frac_bits)

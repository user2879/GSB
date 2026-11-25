"""Deterministic fixed-point integer arithmetic.

The whole ranking pipeline stays in the signed 64-bit integer domain.
numpy.int64 wraps on overflow with identical two's-complement semantics on
Windows and Linux, so cross-platform re-runs produce bit-identical results.
"""
from __future__ import annotations

import numpy as np

WIDTH_TIER = "i64q16"
FRAC_BITS = 16
ONE = np.int64(1 << FRAC_BITS)
MAX_I64 = (1 << 63) - 1
MIN_I64 = -(1 << 63)


def to_q16(value: int) -> np.int64:
    value = int(value)
    scaled = value << FRAC_BITS
    if not MIN_I64 <= scaled <= MAX_I64:
        raise OverflowError("q16 scaling overflow")
    return np.int64(scaled)


def mul_q16(a, b) -> np.int64:
    """Multiply two q16 numbers and renormalize back to q16."""
    product = np.int64(a) * np.int64(b)
    if product >= 0:
        return np.int64(product >> FRAC_BITS)
    # truncation toward zero, matching integer division semantics for scores
    return np.int64(-((-product) >> FRAC_BITS))


def vector_sum_q16(vector) -> np.int64:
    """Deterministic integer sum of a slot vector (ordinary ints), in q16."""
    total = np.int64(0)
    for element in vector:
        total = np.int64(total + np.int64(np.int64(element) << FRAC_BITS))
    return total


def slot_contribution(weight_q16, penalty_q16, value_q16) -> np.int64:
    effective_weight = mul_q16(weight_q16, penalty_q16)
    return np.int64(np.int64(effective_weight) * np.int64(value_q16))


def to_int(value) -> int:
    return int(np.int64(value))

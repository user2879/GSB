"""Unified fixed-point integer engine.

Precision contract (explicit and shared by every node):

* the grid is ``scale = 10 ** k`` for a non-negative integer ``k``;
* all weights and derivation coefficients are quantized with *truncation
  toward zero* (``ROUND_DOWN``); the per-value single-quantization error is an
  integer upper bound of ``0`` (exact) or ``1`` (one quantum at most);
* every multiply/accumulate runs on Python integers; the single division that
  re-enters the digest domain also truncates toward zero;
* digest-domain results saturate to the signed 32-bit range
  ``[INT32_MIN, INT32]`` and report a saturation flag instead of overflowing.

Two error kinds are kept distinct everywhere:

* ``single``  - quantization error introduced by this path's own numbers;
* ``propagated`` - error inherited from already-bounded upstream digests.
"""
from __future__ import annotations

import math
from decimal import ROUND_DOWN, Decimal, localcontext
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .errors import GridError

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def validate_scale(scale: Any, location: str = "session.scale") -> int:
    """Reject anything that is not a positive integer power of 10."""
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise GridError(
            f"scale must be a positive integer power of 10, got {scale!r}",
            location,
            {"scale": scale},
        )
    if scale <= 0:
        raise GridError(
            f"scale must be positive, got {scale}", location, {"scale": scale}
        )
    probe = scale
    while probe > 1:
        if probe % 10 != 0:
            raise GridError(
                f"scale {scale} is not an integer power of 10",
                location,
                {"scale": scale},
            )
        probe //= 10
    return scale


def quant_version(scale: int) -> str:
    """Stable identifier of the quantization grid embedded in receipts."""
    return f"Q32-S{scale}"


def saturate(value: int) -> Tuple[int, bool]:
    """Clip to signed 32-bit, returning ``(clipped_value, saturated)``."""
    if value > INT32_MAX:
        return INT32_MAX, True
    if value < INT32_MIN:
        return INT32_MIN, True
    return value, False


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division with explicit truncation toward zero."""
    if denominator == 0:
        raise ZeroDivisionError("fixed-point divisor must be non-zero")
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        return -quotient
    return quotient


def quantize_scalar(value: Any, scale: int) -> Tuple[int, int]:
    """Quantize one decimal weight/coefficient.

    Returns ``(quantized, single_error_bound)`` where the bound is ``1`` when a
    non-zero fractional quantum was truncated and ``0`` for an exact fit.
    """
    validate_scale(scale)
    with localcontext() as ctx:
        ctx.prec = 50
        dec = Decimal(str(value)) * Decimal(scale)
        q = int(dec.to_integral_value(rounding=ROUND_DOWN))
        residual = dec - Decimal(int(dec))
    single = 1 if residual != 0 else 0
    q, _ = saturate(q)
    return q, single


def quantize_array(values: Sequence[Any], scale: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized quantization for batch weights (numpy implementation).

    Mirrors :func:`quantize_scalar` for sample data that fits float64; returns
    int64 quantized values and uint8 single-error flags.
    """
    validate_scale(scale)
    arr = np.asarray(list(values), dtype=np.float64)
    scaled = arr * float(scale)
    q = np.trunc(scaled).astype(np.int64)
    single = (scaled != np.floor(scaled)).astype(np.uint8)
    q = np.clip(q, INT32_MIN, INT32_MAX)
    return q, single


def leaf_digest(content_hash: int, weight_q: int) -> Tuple[int, bool]:
    """Weighted digest of an independent (non-derived) content path.

    The unsigned 32-bit content hash is treated as a fraction in ``[0, 1)``
    (``hash / 2**32``); the digest stays in the same scale-unit domain as the
    quantized weight.
    """
    numerator = int(content_hash) & 0xFFFFFFFF
    raw = (numerator * int(weight_q)) >> 32
    return saturate(raw)


def derived_digest(
    terms: Sequence[Tuple[int, int]], scale: int
) -> Tuple[int, bool, Dict[str, int]]:
    """Combine upstream digests with quantized coefficients.

    ``terms`` is a sequence of ``(coefficient_q, upstream_digest)`` pairs.
    Returns ``(digest, saturated, error_breakdown)`` where the breakdown exposes
    the propagation component and the single-value component separately, both
    as integer upper bounds in digest (scale-unit) domain.
    """
    validate_scale(scale)
    accumulator = 0
    for coeff_q, digest in terms:
        accumulator += int(coeff_q) * int(digest)
    raw = trunc_div(accumulator, scale)
    digest, saturated = saturate(raw)
    return digest, saturated, {
        "accumulator": accumulator,
        "raw_before_saturate": raw,
    }


def derived_error_bound(
    terms: Sequence[Tuple[int, int, int, int]], scale: int
) -> Dict[str, int]:
    """Integer error upper bound for a derived path.

    Each term contributes ``(coeff_q, coeff_single_err, upstream_digest,
    upstream_bound)``.  The coefficient's own truncation is the *single-value*
    component; weighted upstream bounds form the *propagation* component.  The
    final toward-zero division contributes at most one quantum, added once.
    """
    single_num = 0
    propagation_num = 0
    for coeff_q, coeff_single, upstream_digest, upstream_bound in terms:
        single_num += int(coeff_single) * abs(int(upstream_digest))
        propagation_num += abs(int(coeff_q)) * int(upstream_bound)
    total_num = single_num + propagation_num
    propagated = math.ceil(propagation_num / scale)
    single = math.ceil(single_num / scale)
    bound = math.ceil(total_num / scale) + 1  # +1 for the truncating division
    bound, _ = saturate(bound)
    return {
        "single": single,
        "propagated": propagated,
        "bound": bound,
    }


def leaf_error_bound(weight_single: int) -> Dict[str, int]:
    """Error upper bound for a leaf path: only its own weight quantization."""
    single = 1 if weight_single else 0
    return {"single": single, "propagated": 0, "bound": single}


def describe_contract(scale: int) -> Dict[str, Any]:
    """Human/machine readable statement of the active precision contract."""
    validate_scale(scale)
    return {
        "scale": scale,
        "quant_version": quant_version(scale),
        "truncation": "toward_zero",
        "saturation": "signed_int32",
        "int32_min": INT32_MIN,
        "int32_max": INT32_MAX,
        "integer_ops": [
            "quantize_decimal",
            "multiply",
            "accumulate",
            "trunc_divide",
            "saturate",
        ],
    }


def combine_error_rows(rows: List[Dict[str, int]]) -> Dict[str, int]:
    """Sum per-path error rows for session-level summaries (order stable)."""
    totals = {"single": 0, "propagated": 0, "bound": 0}
    for row in rows:
        for key in totals:
            totals[key] += int(row[key])
    return totals

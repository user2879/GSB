"""Explicit integer fixed-point primitives.

Representation
--------------
Every value is a signed 32-bit integer holding fixed-point ulps at a declared
scale exponent ``e``: real value ``v`` is represented by ``x`` where
``v = x / 10**e``.  Scale exponents are restricted to 0..6 (session scale) and
stage-local scales drawn from the same allowed set.

Declared arithmetic rules (used everywhere, no float anywhere on the path)
--------------------------------------------------------------------------
* Rounding direction for every integer division is "round half away from zero"
  (see :func:`div_half_away`).  Ties never depend on the sign of the result.
* Overflow is saturated to the signed 32-bit envelope
  ``[-2**31, 2**31-1]`` (see :func:`sat32`) and the saturated row is reported
  separately by callers; it never wraps.
* Nulls are represented by ``None`` and propagated: any operation receiving a
  ``None`` operand returns ``None`` without touching the accumulator.
"""

from __future__ import annotations

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
ALLOWED_SCALE_EXPS = (0, 1, 2, 3, 4, 5, 6)
POW10 = [10**i for i in range(7)]

# Fixed point precision (binary) used internally by the integer ln routine.
_LOG_FRAC_BITS = 30
# ln(2) and ln(10) at _LOG_FRAC_BITS precision (verified against math in tests).
_LN2_F = 744261118  # round-half-even(ln2 * 2**30)
_LN10_F = 2472381918  # round-half-even(ln10 * 2**30)


class ScaleError(ValueError):
    """Raised when a scale exponent is outside the allowed set."""


def check_scale(exp: int) -> None:
    if not isinstance(exp, int) or isinstance(exp, bool) or exp not in ALLOWED_SCALE_EXPS:
        raise ScaleError(f"scale exponent {exp!r} not in {ALLOWED_SCALE_EXPS}")


def sat32(x: int) -> int:
    """Saturate an arbitrary-precision integer to the signed 32-bit envelope."""
    if x > INT32_MAX:
        return INT32_MAX
    if x < INT32_MIN:
        return INT32_MIN
    return x


def div_half_away(num: int, den: int) -> int:
    """Signed integer division, rounding half away from zero.

    ``den`` must be positive.  Equivalent to ``trunc(num/den + sign/2)`` but
    implemented without floating point.
    """
    if den <= 0:
        raise ValueError("divisor must be positive")
    q, r = divmod(abs(num), den)
    if r * 2 >= den:
        q += 1
    return -q if num < 0 else q


def rescale_int(x: int | None, from_exp: int, to_exp: int) -> int | None:
    """Rescale fixed-point ulps between scale exponents, rounding/saturating."""
    if x is None:
        return None
    check_scale(from_exp)
    check_scale(to_exp)
    if to_exp == from_exp:
        return sat32(x)
    if to_exp > from_exp:
        return sat32(x * POW10[to_exp - from_exp])
    return sat32(div_half_away(x, POW10[from_exp - to_exp]))


def mul_quant(x: int | None, y: int | None, in_exp: int, out_exp: int):
    """Fixed-point multiply of two values sharing ``in_exp`` to ``out_exp``.

    Returns ``(result, saturated)``; saturated is True when the product hit the
    32-bit envelope after rescaling.
    """
    if x is None or y is None:
        return None, False
    raw = x * y  # exact arbitrary precision: real product * 10**(2*in_exp)
    scaled = rescale_int(raw, 2 * in_exp, out_exp) if 2 * in_exp <= 6 else \
        sat32(div_half_away(raw, POW10[2 * in_exp - out_exp]))
    saturated = (scaled == INT32_MAX and raw > 0) or (scaled == INT32_MIN and raw < 0)
    return scaled, saturated


def bps_ratio(numer: int, denom: int) -> int:
    """Return round-half-away ``numer/denom`` expressed in basis points (1e-4)."""
    if denom == 0:
        return 0
    return div_half_away(numer * 10000, denom)


def _ilog_fixed(n: int) -> int:
    """ln(n) for positive integer n, in 2**_LOG_FRAC_BITS fixed point."""
    if n <= 0:
        raise ValueError("ln undefined for non-positive input")
    bits = n.bit_length()
    # u = n / 2**(bits-1) in [1, 2); represent u at FRAC precision.
    u_f = (n << _LOG_FRAC_BITS) >> (bits - 1)
    one_f = 1 << _LOG_FRAC_BITS
    t_f = u_f - one_f  # t = u - 1 in [0, 1)
    # ln(1+t) = t - t^2/2 + t^3/3 - ... entirely in FRAC fixed point.
    total = 0
    term = t_f
    k = 1
    while term != 0 and k <= 128:
        add = div_half_away(term, k)
        total = total + add if k % 2 == 1 else total - add
        term = div_half_away(term * t_f, one_f)
        k += 1
    return (bits - 1) * _LN2_F + total


def ilog(x: int | None, in_exp: int, base: str, out_exp: int) -> int | None:
    """Integer fixed-point logarithm of a positive fixed-point value.

    Non-positive inputs and nulls yield ``None`` (null propagation rule).
    ``base`` is one of ``"e"``, ``"2"``, ``"10"``.
    """
    if x is None:
        return None
    if x <= 0:
        return None
    check_scale(in_exp)
    check_scale(out_exp)
    ln_x = _ilog_fixed(x)
    ln_scale = _ilog_fixed(POW10[in_exp])
    ln_value = ln_x - ln_scale  # ln(real value) at FRAC precision
    if base == "2":
        ln_value = div_half_away(ln_value << _LOG_FRAC_BITS, _LN2_F)
    elif base == "10":
        ln_value = div_half_away(ln_value << _LOG_FRAC_BITS, _LN10_F)
    elif base != "e":
        raise ValueError(f"unsupported log base {base!r}")
    # Convert from 2**FRAC precision to decimal scale 10**out_exp.
    scaled = div_half_away(ln_value * POW10[out_exp], 1 << _LOG_FRAC_BITS)
    return sat32(scaled)

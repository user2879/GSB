"""Fixed-point quantization using integer arithmetic only.

A session scale factor ``s = 10 ** k`` with ``k in 0..3`` quantizes observed
values and environment weights into signed 64-bit integers.

Explicit rules:

* Truncation direction: towards zero (Python ``int()`` semantics), applied
  symmetrically to positive and negative numbers.
* Saturation: values whose exact scaled magnitude exceeds the signed 64-bit
  range saturate to ``INT64_MAX`` / ``INT64_MIN`` and flag saturation.
* Multiplication / division use Python arbitrary precision integers; no float
  operation participates in a quantized comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import FixedPointError

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
ALLOWED_POWERS = (0, 1, 2, 3)


@dataclass(frozen=True)
class Quantized:
    """A quantized integer plus bookkeeping for error accounting."""

    q: int
    scale: int
    saturated: bool
    truncation_error_ulp: int  # signed remainder in quantized units, towards zero

    @property
    def error_ulp(self) -> int:
        """Absolute truncation remainder, always in ``[0, 1)`` ulp => 0 or ... ."""
        return abs(self.truncation_error_ulp)


class FixedPoint:
    """Session-wide fixed-point context for one scale power ``k``."""

    def __init__(self, power: int) -> None:
        if not isinstance(power, int) or isinstance(power, bool):
            raise FixedPointError(
                "scale power must be an integer", field="scale_power"
            )
        if power not in ALLOWED_POWERS:
            raise FixedPointError(
                f"scale power must be one of {ALLOWED_POWERS}, got {power}",
                field="scale_power",
            )
        self.power = power
        self.scale = 10**power
        # One unit in the last place, expressed as an integer quantized step.
        self.ulp = 1

    # -- quantization -----------------------------------------------------
    def quantize(self, raw: int | float) -> Quantized:
        """Quantize a raw numeric value to a saturated signed 64-bit int."""
        if isinstance(raw, bool):
            raise FixedPointError("boolean is not a quantizable value", field="value")
        if not isinstance(raw, (int, float)):
            raise FixedPointError(
                f"cannot quantize type {type(raw).__name__}", field="value"
            )
        if isinstance(raw, float):
            # Exact rational scaling then truncation towards zero; no binary
            # float artifact enters the quantized result.
            exact = _trunc_float(raw, self.scale)
        else:
            exact = raw * self.scale
        saturated = False
        if exact > INT64_MAX:
            q, saturated = INT64_MAX, True
        elif exact < INT64_MIN:
            q, saturated = INT64_MIN, True
        else:
            q = exact
        remainder = exact - q if saturated else 0
        return Quantized(q=q, scale=self.scale, saturated=saturated,
                         truncation_error_ulp=remainder)

    def dequantize(self, q: int) -> str:
        """Return an exact decimal text representation of a quantized int."""
        sign = "-" if q < 0 else ""
        magnitude = abs(q)
        whole, frac = divmod(magnitude, self.scale)
        if self.scale == 1:
            return f"{sign}{whole}"
        return f"{sign}{whole}.{str(frac).zfill(self.power)}"

    # -- integer arithmetic helpers --------------------------------------
    def scaled_weight(self, weight: int) -> int:
        """Quantize an integer environment weight (weights are integers)."""
        return self.quantize(weight).q

    def mul(self, a: int, b: int) -> int:
        """Integer multiply with signed 64-bit saturation."""
        return _saturate(a * b)

    def div_trunc(self, numerator: int, denominator: int) -> int:
        """Integer division truncating towards zero with saturation guard."""
        if denominator == 0:
            raise FixedPointError("division by zero", field="denominator")
        # Python // floors; emulate C-style truncation towards zero.
        q = abs(numerator) // abs(denominator)
        if (numerator < 0) != (denominator < 0):
            q = -q
        return _saturate(q)

    # -- error bounds -----------------------------------------------------
    def single_key_error_ulp(self) -> int:
        """Worst-case single-key quantization error: 1 ulp."""
        return 1

    def comparison_bound_ulp(self, tolerance_raw: int) -> int:
        """Synthetic comparison bound in quantized units.

        Two candidates each carry at most 1 ulp truncation error, plus the
        declared integer tolerance (converted to quantized units).
        """
        tol_q = int(tolerance_raw) * self.scale
        return 2 * self.ulp + tol_q

    def view_aggregate_bound_ulp(self, key_count: int, tolerance_raw: int) -> int:
        """View-level aggregate upper bound over all compared keys."""
        return key_count * self.comparison_bound_ulp(tolerance_raw)


def _trunc_float(raw: float, scale: int) -> int:
    """Scale a decimal float and truncate towards zero.

    The float is read through its shortest decimal text (``str``) via
    :class:`decimal.Decimal`, so the declared decimal value is quantized
    exactly rather than its binary floating-point approximation.
    """
    from decimal import ROUND_DOWN, Decimal

    scaled = Decimal(str(raw)) * scale
    return int(scaled.to_integral_value(rounding=ROUND_DOWN))


def _saturate(value: int) -> int:
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return value

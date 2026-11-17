"""Fixed point quantization infrastructure.

Conventions (explicit, reproducible, integer only at decision time):

* Grid basis ``G = 10000``: every rule ratio is expressed as an integer number
  of per-10000 grid cells (万分比), ``0 <= x <= G``.
* Session scale exponent ``k`` is an integer power of ten, ``0 <= k <= 6``,
  giving scale factor ``S = 10 ** k`` fixed point units per full scale.
* Quantization truncates toward zero, which for non negative values is a floor:
  ``q(x) = floor(x * S / G)``.
* The quantized ratio mapped back to the per-10000 grid is
  ``w_q = q * G / S`` with ``0 <= x - w_q < delta`` where the fixed point grid
  step in per-10000 units is ``delta = G / S``.
* Bucket comparisons never use floats. A bucket ``b`` (integer in ``0..G``)
  falls under ratio ``q`` iff ``b * S < q * G`` (cross multiplication).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .errors import E_GRID, GrayError

GRID_BASIS = 10000
MAX_EXPONENT = 6


def scale_from_exponent(exponent: int) -> int:
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise GrayError(E_GRID, "scale exponent must be an integer 0..6",
                        field_path="scale_exponent", details={"got": repr(exponent)})
    if exponent < 0 or exponent > MAX_EXPONENT:
        raise GrayError(E_GRID, "scale exponent out of range 0..6",
                        field_path="scale_exponent", line=exponent,
                        details={"allowed": f"0..{MAX_EXPONENT}"})
    return 10 ** exponent


def quantize(value: int, scale: int) -> int:
    """Return ``floor(value * scale / GRID_BASIS)`` (truncation toward zero)."""
    return (int(value) * int(scale)) // GRID_BASIS


def grid_step(scale: int) -> Fraction:
    """Fixed point grid step expressed in per-10000 grid units."""
    return Fraction(GRID_BASIS, scale)


@dataclass(frozen=True)
class QuantizedRatio:
    raw: int                      # original per-10000 integer ratio
    scale: int                    # S = 10**k
    fixed: int                    # q = floor(raw * S / G), units per full scale S
    grid_basis: int = GRID_BASIS

    @property
    def effective_per_10000(self) -> Fraction:
        """Quantized ratio mapped back to the per-10000 grid."""
        return Fraction(self.fixed * self.grid_basis, self.scale)

    @property
    def truncation_error(self) -> Fraction:
        """Actual truncation error in per-10000 grid units (>= 0)."""
        return Fraction(self.raw) - self.effective_per_10000

    @property
    def error_bound(self) -> Fraction:
        """Worst case absolute error bound in per-10000 grid units."""
        return grid_step(self.scale)

    @property
    def threshold_bucket(self) -> Fraction:
        """Exclusive bucket threshold on the ``0..G`` hash bucket axis."""
        return self.effective_per_10000

    def contains_bucket(self, bucket: int) -> bool:
        """Pure integer comparison: bucket is inside this ratio region."""
        return bucket * self.scale < self.fixed * self.grid_basis

    def to_report(self) -> dict:
        eff = self.effective_per_10000
        return {
            "raw_per_10000": self.raw,
            "scale_exponent": len(str(self.scale)) - 1,
            "scale_factor": self.scale,
            "fixed_point": self.fixed,
            "effective_per_10000": _fraction_str(eff),
            "truncation_direction": "floor_toward_zero",
            "truncation_error_per_10000": _fraction_str(self.truncation_error),
            "error_bound_per_10000": _fraction_str(self.error_bound),
            "bucket_threshold_per_10000": _fraction_str(self.threshold_bucket),
        }


def complement_fixed(quantized: int, scale: int) -> int:
    """Stable counterpart so gray + stable partition ``[0, scale)`` exactly."""
    return scale - quantized


def independent_sum_deviation(gray_raw: int, stable_raw: int, scale: int) -> int:
    """Deviation in fixed point units when both sides are quantized independently.

    Independent floor quantization yields ``q_g + q_s`` that may differ from the
    full scale ``S`` by at most one fixed point unit (each floor only removes a
    fraction of a unit). Routing instead uses :func:`complement_fixed`, which
    forces the partition to sum to ``S`` and shifts that single unit entirely
    onto the boundary.
    """
    qg = quantize(gray_raw, scale)
    qs = quantize(stable_raw, scale)
    return scale - (qg + qs)


def _fraction_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"

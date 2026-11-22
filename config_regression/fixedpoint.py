"""Integer fixed-point quantization and deterministic error bounds.

Precision convention
--------------------
* The scale factor is ``10 ** scale_exponent`` (a positive integer power of
  ten).  Real value ``x`` is represented as the integer
  ``q = trunc_toward_zero(x * scale)``.
* Truncation is therefore *toward zero* (not floor), so the per-value
  quantization residual always satisfies ``|x - q/scale| < 1/scale``.
* All arithmetic on quantized values uses Python arbitrary-precision integers
  and is then saturated into the signed 64-bit range.  Saturation clamps to
  ``INT64_MIN`` / ``INT64_MAX``; a saturated operation is flagged in reports.
* Fixed-point multiplication of ``a`` (scale s) by coefficient ``c`` (also
  scale s) producing a value at scale s is
  ``mul(a, c) = sat(trunc_toward_zero(a * c / s))``.

Every bound is expressed in *integer quantized units*, so the entire bound
computation is integer-only and reproducible bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _trunc_div(a: int, b: int) -> int:
    """Integer division with truncation toward zero (b > 0)."""

    assert b > 0
    q = abs(a) // b
    return q if a >= 0 else -q


def saturate(value: int) -> tuple[int, bool]:
    """Clamp *value* into the signed 64-bit range, reporting saturation."""

    if value > INT64_MAX:
        return INT64_MAX, True
    if value < INT64_MIN:
        return INT64_MIN, True
    return value, False


def validate_scale(scale_exponent: int) -> int:
    if not isinstance(scale_exponent, int) or isinstance(scale_exponent, bool):
        raise ValueError("scale_exponent must be an integer")
    if not 0 <= scale_exponent <= 12:
        raise ValueError("scale_exponent must be within [0, 12]")
    return 10 ** scale_exponent


@dataclass(frozen=True)
class FixedPoint:
    scale_exponent: int

    @property
    def scale(self) -> int:
        return 10 ** self.scale_exponent

    def quantize(self, real: float) -> int:
        """Quantize a real number with truncation toward zero + saturation."""

        q, _ = saturate(_raw_quantize(real, self.scale))
        return q

    def mul(self, a: int, b: int) -> tuple[int, bool]:
        """Fixed-point multiply: a*b/scale, truncate toward zero, saturate."""

        return saturate(_trunc_div(a * b, self.scale))

    def add(self, a: int, b: int) -> tuple[int, bool]:
        return saturate(a + b)


def _raw_quantize(real: float, scale: int) -> int:
    # Round-half handling is irrelevant: truncation toward zero is the
    # declared rule.  Use a tiny epsilon only to cancel binary float noise
    # for values that are exact decimals (e.g. 0.1 * 1000).
    scaled = real * scale
    if scaled >= 0:
        return int(math.floor(scaled + 1e-9)) if abs(scaled - round(scaled)) < 1e-9 else int(math.floor(scaled))
    return -int(math.floor(-scaled + 1e-9)) if abs(-scaled - round(-scaled)) < 1e-9 else -int(math.floor(-scaled))


def quantize_value(real: float, scale: int) -> int:
    """Public quantizer: truncate toward zero, then saturate to int64."""

    q, _ = saturate(_raw_quantize(real, scale))
    return q


def coefficient_overbound(c_int: int, scale: int) -> int:
    """Integer over-approximation of ``|c| / scale``.

    Returns the smallest integer ``p`` with ``p >= |c| / scale``, computed
    using integer arithmetic (ceil division).  Used so that error-bound
    propagation stays purely integer.
    """

    return (abs(c_int) + scale - 1) // scale


@dataclass
class BoundComponents:
    """Error-bound components for one config item (integer units).

    ``self_bound``      – quantization residual of the item's own value (1).
    ``propagated``      – sum of propagated child residuals.
    ``truncation``      – residuals introduced by fixed-point multiplications.
    ``total``           – saturated cumulative upper bound (|error| <= total).
    ``saturated``       – whether propagation arithmetic hit int64 saturation.
    """

    self_bound: int
    propagated: int
    truncation: int
    total: int
    saturated: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "self_bound": self.self_bound,
            "propagated": self.propagated,
            "truncation": self.truncation,
            "total": self.total,
            "saturated": self.saturated,
        }


def compute_item_bounds(items: dict[str, dict], order: list[str],
                        quantized_values: dict[str, int],
                        quantized_edges: dict[tuple[str, str], int],
                        scale: int) -> dict[str, BoundComponents]:
    """Propagate integer error bounds along one version's derivation DAG.

    ``items[key]["derived_from"]`` is a list of parent keys; edges carry a
    quantized coefficient.  Bounds are evaluated in the supplied topological
    order so parents are always resolved before children.

    Recurrence (all integers, quantized units)::

        p_i = ceil_overbound(|coeff_i| / scale)
        propagated  = sum( (p_i * e_parent + e_parent + 2) )
        truncation  = number_of_parents            (one trunc residual each)
        e_item      = sat(self_bound + propagated + truncation)

    The ``+ 2`` term keeps the over-bound sound despite replacing the real
    coefficient by its integer over-approximation (one unit for the rounded
    coefficient times the parent value, one unit for the parent residual).
    """

    bounds: dict[str, BoundComponents] = {}
    for key in order:
        parents = items[key].get("derived_from", [])
        if not parents:
            bounds[key] = BoundComponents(1, 0, 0, 1, False)
            continue
        propagated = 0
        truncation = 0
        for parent in parents:
            coeff = quantized_edges[(parent, key)]
            p_i = coefficient_overbound(coeff, scale)
            e_parent = bounds[parent].total
            propagated += p_i * e_parent + e_parent + 2
            truncation += 1
        total, saturated = saturate(1 + propagated + truncation)
        bounds[key] = BoundComponents(1, propagated, truncation, total, saturated)
    return bounds


def within_bound(delta_int: int, bound_a: int, bound_b: int,
                 extra_residual: int) -> tuple[bool, dict[str, int]]:
    """Decide whether a cross-version delta is explained by quantization.

    ``|delta| <= bound_a + bound_b + extra_residual``.  ``extra_residual`` is
    2 for two independently quantized real values, 0 when comparing integers
    that were never re-quantized (e.g. fixed-point derived outputs).
    """

    limit = bound_a + bound_b + extra_residual
    components = {
        "abs_delta": abs(delta_int),
        "baseline_bound": bound_a,
        "candidate_bound": bound_b,
        "cross_version_residual": extra_residual,
        "limit": limit,
    }
    return abs(delta_int) <= limit, components

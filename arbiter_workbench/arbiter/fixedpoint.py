"""Fixed-point integer quantization and explicit error-bound attribution.

Session scale factor ``s`` is an exponent in [0, 3], so the quantum is
``U = 10**-s`` and one stored unit equals ``U`` in real value units.

Rules (all arithmetic on integers, no floats on the data path):

* Truncation direction: round toward zero (Python ``//`` on same-sign operands
  is floor, so an explicit sign-aware truncation helper is used).
* Overflow saturation: quantized values saturate at signed 64-bit bounds
  ``[-2**63, 2**63 - 1]`` and the saturation is recorded.
* Error bounds are kept in the same fixed-point unit and split into three
  contributions: quantization truncation, interval boundary attribution,
  weight normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal, localcontext

from .errors import ErrorContext, QuantError

MIN_INT64 = -(2**63)
MAX_INT64 = 2**63 - 1
ALLOWED_EXPONENTS = (0, 1, 2, 3)


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    """Integer division that truncates toward zero (C-style)."""
    if denominator == 0:
        raise QuantError("division by zero in fixed-point math", ErrorContext())
    quotient = abs(numerator) // abs(denominator)
    if (numerator < 0) != (denominator < 0):
        return -quotient
    return quotient


@dataclass(frozen=True)
class Quantized:
    q: int
    saturated: bool
    truncation_units: int  # |original*10^s - q|, in stored units


class FixedPoint:
    def __init__(self, exponent: int):
        if exponent not in ALLOWED_EXPONENTS:
            raise QuantError(
                "scale exponent must be one of 0..3",
                ErrorContext(field=f"scale={exponent}"),
            )
        self.exponent = exponent
        self.factor = 10**exponent
        # quantum as Decimal for exact decimal input parsing
        self.quantum = Decimal(1).scaleb(-exponent)

    def quantize(self, raw_value) -> Quantized:
        """Quantize a number/string/Decimal toward zero, saturating at int64."""
        with localcontext() as ctx:
            ctx.prec = 40
            dec = raw_value if isinstance(raw_value, Decimal) else Decimal(str(raw_value))
            scaled = (dec * self.factor).to_integral_value(rounding=ROUND_DOWN)
            full = int(scaled)
        saturated = full < MIN_INT64 or full > MAX_INT64
        q = max(MIN_INT64, min(MAX_INT64, full))
        truncation = abs(full - q) if saturated else 0
        # truncation toward zero residual against the exact scaled value
        exact_scaled = dec * self.factor
        residual = int(
            abs(exact_scaled - Decimal(full)).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )
        return Quantized(q=q, saturated=saturated, truncation_units=residual + truncation)

    def dequantize(self, q: int) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = 40
            return Decimal(q) / Decimal(self.factor)

    # ---- integer-only arithmetic on quantized values --------------------
    def add(self, a: int, b: int) -> int:
        return self._saturate(a + b)

    def sub(self, a: int, b: int) -> int:
        return self._saturate(a - b)

    def mul_scalar(self, a: int, scalar: int) -> int:
        return self._saturate(a * scalar)

    def mean(self, values: list[int]) -> int:
        if not values:
            raise QuantError("mean of empty value set", ErrorContext())
        total = 0
        for value in values:
            total = self._saturate(total + value)
        return trunc_div_toward_zero(total, len(values))

    @staticmethod
    def _saturate(value: int) -> int:
        if value < MIN_INT64:
            return MIN_INT64
        if value > MAX_INT64:
            return MAX_INT64
        return value


@dataclass(frozen=True)
class ErrorContribution:
    quantization_truncation: int  # units
    interval_boundary: int       # units
    weight_normalization: int    # units

    def total(self) -> int:
        return (
            self.quantization_truncation
            + self.interval_boundary
            + self.weight_normalization
        )


def truncation_bound(fp: FixedPoint, contributor_count: int) -> int:
    """Each operand contributes < 1 stored unit of toward-zero truncation.

    A linear combination of ``contributor_count`` operands therefore has a
    truncation bound of ``contributor_count`` stored units (worst case).
    """
    return contributor_count


def interval_boundary_bound(fp: FixedPoint) -> int:
    """Membership of an interval endpoint is decided on quantized integers.

    A real endpoint can fall at most one stored unit away from its quantized
    counterpart, so inclusion/exclusion of a boundary value costs at most one
    unit in the resulting midpoint.
    """
    return 1


def normalize_weights(weights: list[int], fp: FixedPoint) -> tuple[list[int], int]:
    """Normalize positive integer weights to fixed-point fractions summing
    exactly to ``fp.factor`` (largest-remainder style, deterministic order).

    Returns the normalized weights (in stored units, sum == factor) and the
    weight-normalization error bound in stored units. The largest-remainder
    allocation guarantees every individual rounding residual < 1 unit; the
    aggregate residual is redistributed so the normalization error bound is
    ``len(weights) - 1`` units.
    """
    if not weights or any(w <= 0 for w in weights):
        raise QuantError(
            "weights must be positive integers", ErrorContext(field="weights")
        )
    total = sum(weights)
    raw = [w * fp.factor for w in weights]
    floors = [trunc_div_toward_zero(r, total) for r in raw]
    remainders = [r - floor * total for r, floor in zip(raw, floors)]
    allocated = list(floors)
    leftover = fp.factor - sum(allocated)
    # deterministic order: largest remainder first, then original index
    order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))
    for idx in order[:leftover]:
        allocated[idx] += 1
    assert sum(allocated) == fp.factor
    bound = max(0, len(weights) - 1)
    return allocated, bound


def per_key_error_bound(
    fp: FixedPoint,
    policy: str,
    env_count: int,
) -> ErrorContribution:
    """Per-key comparison/judgment error bound split into three contributions."""
    quant = truncation_bound(fp, env_count)
    boundary = interval_boundary_bound(fp) if policy == "interval_midpoint" else 0
    _, norm_bound = normalize_weights([1] * env_count, fp)
    if policy != "weighted_majority":
        norm_bound = 0
    return ErrorContribution(
        quantization_truncation=quant,
        interval_boundary=boundary,
        weight_normalization=norm_bound,
    )


def view_aggregate_bound(per_key_bounds: list[ErrorContribution]) -> ErrorContribution:
    """View-level aggregate upper bound: uniform (max) norm across keys."""
    if not per_key_bounds:
        return ErrorContribution(0, 0, 0)
    return ErrorContribution(
        quantization_truncation=max(b.quantization_truncation for b in per_key_bounds),
        interval_boundary=max(b.interval_boundary for b in per_key_bounds),
        weight_normalization=max(b.weight_normalization for b in per_key_bounds),
    )

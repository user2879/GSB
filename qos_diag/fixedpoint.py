from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import SpecError


MAX_FIXED_Q = (1 << 63) - 1


@dataclass(frozen=True)
class FixedPoint:
    """Decimal fixed-point configuration.

    scale=0..3 means Q=10**scale fractional decimal digits.  Non-negative
    quantization uses floor/truncation toward zero (identical for non-negative
    numbers).  Multiplication/addition saturate at MAX_FIXED_Q instead of
    wrapping.  Balances are additionally clamped to the declared burst cap.
    """

    scale: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.scale, int) or isinstance(self.scale, bool):
            raise SpecError("scale must be an integer from 0 to 3", object_id="$config", field="scale")
        if self.scale not in (0, 1, 2, 3):
            raise SpecError("scale must be an integer from 0 to 3", object_id="$config", field="scale")

    @property
    def q(self) -> int:
        return 10**self.scale

    def quantize_integer(self, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SpecError("fixed-point input must be an integer", field="value")
        return saturating_mul(value, self.q)[0]

    def floor_ratio(self, numerator: int, denominator: int) -> tuple[int, int]:
        if denominator <= 0:
            return 0, 0
        result = (self.q * numerator) // denominator
        remainder = (self.q * numerator) % denominator
        return result, remainder

    def error_bounds(self) -> dict[str, Any]:
        q = self.q
        return {
            "scale": self.scale,
            "q": q,
            "quantum_tokens": f"1/{q}" if q != 1 else "1",
            "single_event_bound_q_exclusive": 1,
            "single_event_bound_tokens": f"1/{q}" if q != 1 else "0",
            "window_aggregation_bound_q_formula": "judged_or_quantized_terms + floor_divisions",
            "three_stage_chain_bound_q_formula": "sum(stage_window_bounds)",
            "truncation_direction": "floor for non-negative values (toward zero)",
            "overflow_rule": f"saturate_at_{MAX_FIXED_Q}",
            "observed_integer_input_error_q": 0,
        }


def saturating_mul(left: int, right: int) -> tuple[int, bool]:
    value = left * right
    if value > MAX_FIXED_Q:
        return MAX_FIXED_Q, True
    if value < -MAX_FIXED_Q:
        return -MAX_FIXED_Q, True
    return value, False


def saturating_add(left: int, right: int) -> tuple[int, bool]:
    value = left + right
    if value > MAX_FIXED_Q:
        return MAX_FIXED_Q, True
    if value < -MAX_FIXED_Q:
        return -MAX_FIXED_Q, True
    return value, False


def classify_difference(observed_abs_q: int, tolerance_q: int, uncertainty_q: int) -> str:
    """Classify a fixed-point discrepancy.

    Returns one of ``exceeds_tolerance``, ``within_tolerance`` or
    ``indistinguishable``.  The uncertainty is a non-strict upper bound on
    measurement/quantization error in q-units.
    """

    observed_abs_q = abs(int(observed_abs_q))
    tolerance_q = max(0, int(tolerance_q))
    uncertainty_q = max(0, int(uncertainty_q))
    if observed_abs_q - uncertainty_q > tolerance_q:
        return "exceeds_tolerance"
    if observed_abs_q + uncertainty_q <= tolerance_q:
        return "within_tolerance"
    return "indistinguishable"

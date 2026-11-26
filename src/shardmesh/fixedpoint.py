"""Integer fixed-point water-level arithmetic.

Supported scales are explicit.  Non-negative division truncates toward zero,
which is equivalent to floor for the quantities used here.  Multiplication is
saturated at :data:`SATURATION` rather than wrapping or raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import spec_error

SUPPORTED_SCALES = (1, 10, 100, 1000, 10000)
SATURATION = 2**63 - 1


def validate_scale(scale: Any, index: int | None = None, field_name: str = "scale") -> int:
    if not isinstance(scale, int) or isinstance(scale, bool):
        raise spec_error(index, field_name, "scale must be an integer", category="global")
    if scale not in SUPPORTED_SCALES:
        raise spec_error(
            index,
            field_name,
            f"scale {scale!r} is not in supported set {SUPPORTED_SCALES}",
            category="global",
        )
    return scale


def saturated_mul(left: int, right: int) -> tuple[int, bool]:
    value = left * right
    if value > SATURATION:
        return SATURATION, True
    if value < -SATURATION:
        return -SATURATION, True
    return value, False


def to_fixed_units(units: int, scale: int) -> int:
    value, _ = saturated_mul(max(0, int(units)), scale)
    return value


@dataclass(frozen=True)
class QuantizedRatio:
    q: int
    numerator: int
    denominator: int
    scale: int
    remainder: int
    raw_scaled_numerator: int
    saturated: bool

    @property
    def ulp_error_num(self) -> int:
        """Numerator of the truncation error in fixed-point q units."""
        return self.remainder

    @property
    def ulp_error_den(self) -> int:
        return max(1, self.denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "scale": self.scale,
            "remainder": self.remainder,
            "error_q_num": self.ulp_error_num,
            "error_q_den": self.ulp_error_den,
            "error_q_bound": "less_than_1_ulp",
            "saturated": self.saturated,
            "truncation": "floor_toward_zero_for_nonnegative",
            "overflow": f"saturate_at_{SATURATION}",
        }


def quantize_ratio(numerator: int, denominator: int, scale: int) -> QuantizedRatio:
    if denominator <= 0:
        return QuantizedRatio(SATURATION, numerator, denominator, scale, 0, SATURATION, True)
    raw, saturated_mul_flag = saturated_mul(max(0, numerator), scale)
    q_raw = raw // denominator
    saturated = saturated_mul_flag or q_raw > SATURATION
    q = min(q_raw, SATURATION)
    if saturated:
        remainder = 0
        raw_scaled = SATURATION
    else:
        remainder = raw - q * denominator
        raw_scaled = raw
    return QuantizedRatio(q, numerator, denominator, scale, remainder, raw_scaled, saturated)


def ceil_div(left: int, right: int) -> int:
    if right <= 0:
        raise ValueError("right must be positive")
    return -((-left) // right)

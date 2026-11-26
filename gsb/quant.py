from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Any


ROUNDING = {
    "half_up": ROUND_HALF_UP,
    "half_even": ROUND_HALF_EVEN,
    "down": ROUND_DOWN,
}


def q_scale(decimal_width: int) -> int:
    return 10 ** int(decimal_width)


def quantize_number(value: float | int | str, scale: int, rounding: str = "half_up") -> int:
    """Quantize a decimal config/reference value into an integer fixed-point domain."""
    mode = ROUNDING[rounding]
    quantum = Decimal(1)
    return int((Decimal(str(value)) * scale).quantize(quantum, rounding=mode))


def q_to_float(value_q: int, decimal_width: int) -> float:
    return value_q / q_scale(decimal_width)


def round_ratio(numerator_q: int, ratio_numerator: int, ratio_denominator: int, rounding: str) -> int:
    """Round ``numerator_q * ratio_numerator / ratio_denominator`` in integer arithmetic."""
    if ratio_denominator <= 0:
        raise ValueError("ratio_denominator must be positive")
    sign = -1 if numerator_q * ratio_numerator < 0 else 1
    absolute = abs(numerator_q * ratio_numerator)
    if rounding == "down":
        return sign * (absolute // ratio_denominator)
    quotient, remainder = divmod(absolute, ratio_denominator)
    twice = remainder * 2
    if rounding == "half_up":
        if twice >= ratio_denominator:
            quotient += 1
    elif rounding == "half_even":
        if twice > ratio_denominator or (twice == ratio_denominator and quotient % 2):
            quotient += 1
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"unsupported rounding {rounding}")
    return sign * quotient


def gear_width(config: dict[str, Any], gear: str | None = None) -> int:
    fixed = config["fixed_point"]
    if gear == "coarse":
        return 2
    if gear == "standard":
        return 4
    if gear == "fine":
        return 6
    if gear is None:
        return int(fixed["decimal_width"])
    return int(gear[1:]) if isinstance(gear, str) and gear.startswith("Q") else int(fixed["decimal_width"])

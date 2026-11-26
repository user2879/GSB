from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any

from .errors import InputValidationError


ALLOWED_DECIMAL_WIDTHS = (3, 6, 9, 12)
QUANTIZATION_TIERS = {
    "floor": ROUND_FLOOR,
    "truncate": ROUND_FLOOR,
    "nearest": ROUND_HALF_UP,
    "round_half_up": ROUND_HALF_UP,
    "ceil": ROUND_CEILING,
    "ceiling": ROUND_CEILING,
}
CANONICAL_TIERS = {
    "floor": "floor",
    "truncate": "floor",
    "nearest": "half_up",
    "round_half_up": "half_up",
    "half_up": "half_up",
    "ceil": "ceiling",
    "ceiling": "ceiling",
}


def scale_for_width(width: int) -> int:
    return 10**width


def validate_fixed_point_config(config: dict[str, Any]) -> tuple[int, str]:
    fp = config.get("fixed_point")
    field = "fixed_point"
    if not isinstance(fp, dict):
        raise InputValidationError("fixed_point must be an object", field=field)

    width = fp.get("decimal_width")
    if isinstance(width, bool) or not isinstance(width, int) or width not in ALLOWED_DECIMAL_WIDTHS:
        raise InputValidationError(
            f"fixed_point.decimal_width must be one of {list(ALLOWED_DECIMAL_WIDTHS)}",
            field="fixed_point.decimal_width",
        )

    tier_name = fp.get("quantization_tier")
    if not isinstance(tier_name, str) or tier_name not in QUANTIZATION_TIERS:
        raise InputValidationError(
            "fixed_point.quantization_tier must be one of "
            "floor/truncate/nearest/round_half_up/ceil/ceiling",
            field="fixed_point.quantization_tier",
        )
    return width, CANONICAL_TIERS[tier_name]


def quantize_ratio(numerator: int, denominator: int, width: int, tier: str) -> int:
    """Return numerator/denominator * 10**width as an exact rounded integer."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        raise ValueError("numerator must be nonnegative")
    scale = scale_for_width(width)
    scaled_numerator = numerator * scale
    quotient, remainder = divmod(scaled_numerator, denominator)
    canonical_tier = CANONICAL_TIERS.get(tier, tier)
    if canonical_tier == "floor":
        return quotient
    if canonical_tier == "ceiling":
        return quotient + (1 if remainder else 0)
    if 2 * remainder >= denominator:
        return quotient + 1
    return quotient


def quantize_float(value: float | int | str, width: int, tier: str) -> int:
    """Quantize a JSON number through its shortest stable decimal representation."""

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid fixed-point number: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError("number must be finite")
    scale = Decimal(scale_for_width(width))
    rounding = QUANTIZATION_TIERS[tier]
    return int((decimal_value * scale).to_integral_value(rounding=rounding))


def permille_to_fixed(threshold_permille: int, width: int) -> int:
    return threshold_permille * (10 ** (width - 3))

"""Integer fixed-point arithmetic with explicit truncation and saturation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any

from .errors import ColmvError


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def saturate_int64(value: int) -> int:
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return int(value)


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    if numerator == 0:
        return 0
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


@dataclass(frozen=True)
class FixedPointContract:
    scale_power: int

    def __post_init__(self) -> None:
        if not isinstance(self.scale_power, int) or isinstance(self.scale_power, bool):
            raise ValueError("scale_power must be an integer")
        if not 0 <= self.scale_power <= 18:
            raise ValueError("scale_power must be in [0, 18]")

    @property
    def scale(self) -> int:
        return 10**self.scale_power

    @property
    def version(self) -> str:
        return (
            f"fixed-v1;scale=10^{self.scale_power};"
            "trunc=toward-zero;saturate=int64"
        )

    def quantize(self, value: Any, *, column: str | None = None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ColmvError(
                "E_VIEW",
                f"boolean is not a numeric fixed-point value in column {column!r}",
                field=column,
            )
        try:
            if isinstance(value, int):
                decimal_value = Decimal(value)
            elif isinstance(value, float):
                if value != value or value in (float("inf"), float("-inf")):
                    raise ColmvError(
                        "E_VIEW",
                        f"non-finite numeric value in column {column!r}",
                        field=column,
                    )
                decimal_value = Decimal(value)
            else:
                raise ColmvError(
                    "E_VIEW",
                    f"numeric column {column!r} contains non-numeric value",
                    field=column,
                )
            scaled = decimal_value * self.scale
            return int(scaled.to_integral_value(rounding=ROUND_DOWN))
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ColmvError(
                "E_VIEW",
                f"cannot quantize value in column {column!r}: {exc}",
                field=column,
            ) from exc

    def decimal_from_units(self, units: int) -> str:
        """Render an integer number of 1/scale units as an exact decimal string."""

        sign = "-" if units < 0 else ""
        magnitude = abs(int(units))
        if self.scale_power == 0:
            return f"{sign}{magnitude}"
        whole, fraction = divmod(magnitude, self.scale)
        return f"{sign}{whole}.{fraction:0{self.scale_power}d}"

    def quantization_error_units_fraction(self, value: Any) -> tuple[int, int, str]:
        """Return the exact absolute quantization error in fixed-point units."""

        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            return (0, 1, "0")
        if not isinstance(value, float):
            raise TypeError("fractional quantization error requires a numeric value")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("fractional quantization error requires a finite value")
        scaled = Decimal(value) * self.scale
        truncated = scaled.to_integral_value(rounding=ROUND_DOWN)
        remainder = abs(scaled - truncated)
        sign, digits, exponent = remainder.as_tuple()
        numerator = int("".join(str(digit) for digit in digits)) if digits else 0
        if exponent >= 0:
            numerator *= 10**exponent
            denominator = 1
        else:
            denominator = 10 ** (-exponent)
        decimal_text = format(remainder, "f")
        return (numerator, denominator, decimal_text)

    def contract_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scale_power": self.scale_power,
            "scale": self.scale,
            "quantization": "truncate-toward-zero-at-ingest",
            "division": "truncate-toward-zero",
            "overflow": "saturate-signed-int64",
            "count_representation": "count-scaled-by-scale",
        }


def quantization_error_upper_units(value: Any) -> int:
    """Conservative per-value upper bound in fixed-point units.

    Integer inputs have no quantization error.  Floating inputs are truncated
    toward zero, so the absolute error is strictly below one fixed-point unit;
    one unit is the declared conservative bound.
    """

    if value is None:
        return 0
    if isinstance(value, int) and not isinstance(value, bool):
        return 0
    if isinstance(value, float):
        return 1
    return 0

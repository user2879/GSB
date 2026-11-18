from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
)
import hashlib
import json
from typing import Any

from .errors import MigrationError, fail


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
ROUNDING_MODES = ("toward_zero", "floor", "ceil", "nearest_half_even")
SATURATION_MODES = ("reject", "clamp")


def pow10(exponent: int) -> int:
    if exponent < 0:
        raise ValueError("pow10 exponent must be non-negative")
    return 10**exponent


def _ceil_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _floor_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def parse_decimal(value: Any) -> Decimal:
    """Parse a local database/JSON number without introducing binary float noise.

    Floating point values are interpreted through their shortest decimal
    representation (``str(value)``). This is deterministic across runs and is
    the documented input contract for REAL columns.
    """

    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a numeric value")
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, int):
            result = Decimal(value)
        elif isinstance(value, float):
            result = Decimal(str(value))
        elif isinstance(value, str):
            result = Decimal(value.strip())
        elif isinstance(value, bytes):
            result = Decimal(value.decode("ascii").strip())
        else:
            result = Decimal(str(value))
    except (InvalidOperation, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid numeric value {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite numeric value {value!r}")
    return result


@dataclass(frozen=True)
class QuantizationSpec:
    """Declared fixed-point calibration for one numeric column."""

    scale: int
    truncation: str = "toward_zero"
    saturation: str = "reject"
    min_units: int = INT64_MIN
    max_units: int = INT64_MAX

    def validate(self, *, step_id: str | None = None, field_path: str = "quantization") -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            fail(
                "E_MIGRATE",
                "quantization scale must be an integer from 0 to 9",
                step_id=step_id,
                field_path=f"{field_path}.scale",
            )
        if not 0 <= self.scale <= 9:
            fail(
                "E_MIGRATE",
                "quantization scale must be in the inclusive range 0..9",
                step_id=step_id,
                field_path=f"{field_path}.scale",
            )
        if self.truncation not in ROUNDING_MODES:
            fail(
                "E_MIGRATE",
                f"unsupported truncation direction {self.truncation!r}",
                step_id=step_id,
                field_path=f"{field_path}.truncation",
                supported=list(ROUNDING_MODES),
            )
        if self.saturation not in SATURATION_MODES:
            fail(
                "E_MIGRATE",
                f"unsupported saturation rule {self.saturation!r}",
                step_id=step_id,
                field_path=f"{field_path}.saturation",
                supported=list(SATURATION_MODES),
            )
        if (
            isinstance(self.min_units, bool)
            or isinstance(self.max_units, bool)
            or not isinstance(self.min_units, int)
            or not isinstance(self.max_units, int)
        ):
            fail(
                "E_MIGRATE",
                "saturation bounds must be integer unit counts",
                step_id=step_id,
                field_path=field_path,
            )
        if self.min_units > self.max_units:
            fail(
                "E_MIGRATE",
                "saturation minimum must not exceed maximum",
                step_id=step_id,
                field_path=field_path,
            )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        *,
        step_id: str | None = None,
        field_path: str = "quantization",
        default: "QuantizationSpec | None" = None,
    ) -> "QuantizationSpec":
        if payload is None:
            if default is None:
                fail(
                    "E_MIGRATE",
                    "quantization declaration is required",
                    step_id=step_id,
                    field_path=field_path,
                )
            return default
        if not isinstance(payload, dict):
            fail(
                "E_MIGRATE",
                "quantization must be a JSON object",
                step_id=step_id,
                field_path=field_path,
            )
        allowed = {"scale", "truncation", "saturation", "min_units", "max_units"}
        extra = sorted(set(payload) - allowed)
        if extra:
            fail(
                "E_MIGRATE",
                "unknown quantization field",
                step_id=step_id,
                field_path=f"{field_path}.{extra[0]}",
            )
        if "scale" not in payload:
            fail(
                "E_MIGRATE",
                "quantization.scale is required",
                step_id=step_id,
                field_path=f"{field_path}.scale",
            )
        spec = cls(
            scale=payload["scale"],
            truncation=payload.get(
                "truncation", default.truncation if default is not None else "toward_zero"
            ),
            saturation=payload.get(
                "saturation", default.saturation if default is not None else "reject"
            ),
            min_units=payload.get(
                "min_units", default.min_units if default is not None else INT64_MIN
            ),
            max_units=payload.get(
                "max_units", default.max_units if default is not None else INT64_MAX
            ),
        )
        spec.validate(step_id=step_id, field_path=field_path)
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "truncation": self.truncation,
            "saturation": self.saturation,
            "min_units": self.min_units,
            "max_units": self.max_units,
        }

    def version(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QuantizedValue:
    units: int
    scale: int
    error_units: int
    saturated: bool
    truncation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "scale": self.scale,
            "error_units": self.error_units,
            "saturated": self.saturated,
            "truncation": self.truncation,
        }


def _round_decimal_scaled(exact_scaled: Decimal, mode: str) -> int:
    if mode == "toward_zero":
        return int(exact_scaled)
    if mode == "floor":
        return _floor_decimal(exact_scaled)
    if mode == "ceil":
        return _ceil_decimal(exact_scaled)
    if mode == "nearest_half_even":
        return int(exact_scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
    raise ValueError(f"unsupported rounding mode {mode!r}")


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division truncated toward zero, without float conversion."""

    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def floor_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    return numerator // denominator


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    return -((-numerator) // denominator)


def nearest_half_even_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    quotient, remainder = divmod(abs(numerator), abs(denominator))
    doubled = remainder * 2
    if doubled > abs(denominator) or (
        doubled == abs(denominator) and quotient % 2 == 1
    ):
        quotient += 1
    return sign * quotient


def div_round(numerator: int, denominator: int, mode: str) -> int:
    if mode == "toward_zero":
        return trunc_div(numerator, denominator)
    if mode == "floor":
        return floor_div(numerator, denominator)
    if mode == "ceil":
        return ceil_div(numerator, denominator)
    if mode == "nearest_half_even":
        return nearest_half_even_div(numerator, denominator)
    raise ValueError(f"unsupported rounding mode {mode!r}")


def rescale_units(units: int, from_scale: int, to_scale: int, mode: str) -> int:
    if from_scale == to_scale:
        return units
    if to_scale > from_scale:
        return units * pow10(to_scale - from_scale)
    return div_round(units, pow10(from_scale - to_scale), mode)


def quantize_decimal(
    value: Any,
    spec: QuantizationSpec,
    *,
    step_id: str | None = None,
    field_path: str = "value",
) -> QuantizedValue:
    """Quantize one value to an integer grid with explicit truncation/saturation."""

    spec.validate(step_id=step_id, field_path="quantization")
    try:
        decimal_value = parse_decimal(value)
    except ValueError as exc:
        fail(
            "E_MIGRATE",
            str(exc),
            step_id=step_id,
            field_path=field_path,
        )
    exact_scaled = decimal_value * pow10(spec.scale)
    unclamped = _round_decimal_scaled(exact_scaled, spec.truncation)
    saturated = False
    units = unclamped
    if units < spec.min_units or units > spec.max_units:
        saturated = True
        if spec.saturation == "reject":
            fail(
                "E_MIGRATE",
                "fixed-point result exceeds the declared saturation range",
                step_id=step_id,
                field_path=field_path,
                units=unclamped,
                min_units=spec.min_units,
                max_units=spec.max_units,
            )
        units = min(spec.max_units, max(spec.min_units, units))
    residual = abs(exact_scaled - Decimal(units))
    error_units = 0 if residual == 0 else max(1, _ceil_decimal(residual))
    return QuantizedValue(
        units=units,
        scale=spec.scale,
        error_units=error_units,
        saturated=saturated,
        truncation=spec.truncation,
    )


def fixed_to_decimal_string(units: int, scale: int) -> str:
    sign = "-" if units < 0 else ""
    magnitude = abs(units)
    if scale == 0:
        return f"{sign}{magnitude}"
    factor = pow10(scale)
    whole, fraction = divmod(magnitude, factor)
    return f"{sign}{whole}.{fraction:0{scale}d}"


def normalize_error_terms(
    terms: dict[tuple[str, ...], int] | list[tuple[tuple[str, ...], int]],
    *,
    bound: int,
    residual_path: tuple[str, ...],
) -> tuple[list[dict[str, Any]], int]:
    """Return sorted terms whose integer contributions sum to ``bound``.

    Deterministic ordering is by path tuple, never by insertion/hash order.
    """

    if isinstance(terms, dict):
        items = list(terms.items())
    else:
        items = list(terms)
    merged: dict[tuple[str, ...], int] = {}
    for path, units in items:
        if units < 0:
            raise ValueError("error contribution must be non-negative")
        normalized = tuple(str(part) for part in path)
        merged[normalized] = merged.get(normalized, 0) + int(units)
    total = sum(merged.values())
    if total < bound:
        merged[residual_path] = merged.get(residual_path, 0) + (bound - total)
    elif total > bound:
        bound = total

    ordered = sorted(merged.items(), key=lambda pair: pair[0])
    result: list[dict[str, Any]] = []
    for path, units in ordered:
        share = "0" if bound == 0 else format(Decimal(units) / Decimal(bound), ".12f")
        result.append(
            {
                "path": list(path),
                "path_text": " <- ".join(path),
                "units": units,
                "share": share,
            }
        )
    return result, bound


def comparison_bound_units(before_error: int, after_error: int) -> int:
    return max(0, int(before_error)) + max(0, int(after_error))


def make_migration_error(
    message: str,
    *,
    step_id: str | None,
    field_path: str | None,
    **details: Any,
) -> MigrationError:
    return MigrationError(
        "E_MIGRATE",
        message,
        step_id=step_id,
        field_path=field_path,
        details=details,
    )

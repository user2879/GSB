"""Deterministic fixed-point conversion and integer quantisation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP
import math
from typing import Any


ROUNDING_MODES = {
    "half_even": ROUND_HALF_EVEN,
    "half_up": ROUND_HALF_UP,
    "half_down": ROUND_HALF_DOWN,
}


@dataclass(frozen=True)
class FixedValue:
    units: int
    direction: str


def _direction(decimal_value: Decimal, integral: Decimal) -> str:
    if integral > decimal_value:
        return "up"
    if integral < decimal_value:
        return "down"
    return "exact"


def to_fixed(value: Any, decimal_places: int, rounding: str = "half_even") -> FixedValue:
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"unsupported rounding mode: {rounding}")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError("value is not finite")
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"cannot convert value to fixed point: {value!r}") from exc
    scaled = decimal_value * (Decimal(10) ** decimal_places)
    integral = scaled.to_integral_value(rounding=ROUNDING_MODES[rounding])
    return FixedValue(int(integral), _direction(scaled, integral))


def quantize_units(units: int, quantum: int, rounding: str = "half_even") -> FixedValue:
    """Round fixed integer units to the nearest multiple of ``quantum``."""

    if quantum <= 0:
        raise ValueError("quantization level must be a positive integer")
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"unsupported rounding mode: {rounding}")
    decimal_units = Decimal(int(units))
    scaled = decimal_units / Decimal(int(quantum))
    quotient = scaled.to_integral_value(rounding=ROUNDING_MODES[rounding])
    result = int(quotient) * int(quantum)
    original = Decimal(int(units))
    quantized = Decimal(result)
    if quantized > original:
        direction = "quantized_up"
    elif quantized < original:
        direction = "quantized_down"
    else:
        direction = "exact"
    return FixedValue(result, direction)


def fixed_to_float(units: int, decimal_places: int) -> float:
    return float(Decimal(int(units)) / (Decimal(10) ** decimal_places))


def fixed_to_text(units: int, decimal_places: int) -> str:
    value = Decimal(int(units)) / (Decimal(10) ** decimal_places)
    return f"{value:.{decimal_places}f}"


def combine_directions(first: str, second: str) -> str:
    if second != "exact":
        return second
    return first


def deterministic_standard_normal(row_id: str, column: str) -> tuple[float, str]:
    """Derive a standard normal variate from only row id and column name.

    Two independent SHA-256 uniforms feed a Box-Muller transform.  No runtime global
    random state is read or modified.
    """

    import hashlib

    base = f"{row_id}\x1f{column}".encode("utf-8")
    digest1 = hashlib.sha256(b"noise-u1\x1f" + base).digest()
    digest2 = hashlib.sha256(b"noise-u2\x1f" + base).digest()
    u1_int = int.from_bytes(digest1[:8], "big") + 1
    u1 = u1_int / (2**64 + 1)
    u2_int = int.from_bytes(digest2[:8], "big")
    u2 = u2_int / 2**64
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    source = hashlib.sha256(base).hexdigest()
    return z, source


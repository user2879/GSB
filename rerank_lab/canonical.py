"""Platform-stable canonicalisation, hashing and fixed-point helpers.

The numeric pipeline deliberately avoids binary floating point.  JSON input is
parsed with ``parse_float=Decimal`` and converted to ``Fraction`` where exact
arithmetic is required.  Only the final fixed-point integers and deterministic
decimal strings are serialised.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any


def parse_json_constant(text: str) -> Decimal:
    """Keep JSON NaN/Infinity detectable instead of silently accepting them."""

    return Decimal(text)


def as_fraction(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric feature")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal")
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        # Input JSON normally never reaches this branch because of Decimal parsing.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float")
        return Fraction(Decimal(str(value)))
    if isinstance(value, str):
        try:
            dec = Decimal(value)
        except InvalidOperation as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid numeric string: {value!r}") from exc
        if not dec.is_finite():
            raise ValueError("non-finite numeric string")
        return Fraction(dec)
    raise TypeError(f"unsupported numeric type: {type(value).__name__}")


def is_finite_number(value: Any) -> bool:
    try:
        as_fraction(value)
        return True
    except (TypeError, ValueError, InvalidOperation):
        return False


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def decimal_text(value: Fraction | Decimal | int, places: int = 12) -> str:
    with localcontext() as ctx:
        ctx.prec = max(40, places + 20)
        if isinstance(value, Fraction):
            dec = Decimal(value.numerator) / Decimal(value.denominator)
        elif isinstance(value, Decimal):
            dec = value
        else:
            dec = Decimal(value)
        text = format(dec, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text


def canonical_data(value: Any) -> Any:
    """Convert values to a JSON-safe, type-tagged canonical representation."""

    if isinstance(value, Fraction):
        return {"$fraction": fraction_text(value)}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if is_dataclass(value):
        return canonical_data(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): canonical_data(val)
            for key, val in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_data(item) for item in value]
    if isinstance(value, set):
        return sorted(canonical_data(item) for item in value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return {"$float": str(value)}
        return {"$float": repr(value)}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_cache_key(
    stage: str,
    upstream: str,
    config: Any,
    quantization: Any,
) -> str:
    return digest(
        {
            "stage": stage,
            "upstream": upstream,
            "config": config,
            "quantization": quantization,
        }
    )


def round_half_away(value: Fraction) -> int:
    """Round a rational to an integer, with exact halves away from zero."""

    numerator = value.numerator
    denominator = value.denominator
    if numerator >= 0:
        return (2 * numerator + denominator) // (2 * denominator)
    return -((2 * -numerator + denominator) // (2 * denominator))


def fixed_from_real(value: Fraction, places: int, step: int) -> int:
    scale = 10**places
    return round_half_away(value * scale / step)


def weight_fixed_from_real(value: Fraction, places: int) -> int:
    scale = 10**places
    return round_half_away(value * scale)


def score_scale_text(places: int, step: int) -> str:
    return decimal_text(Fraction(step, 10**places), places=places + 4)


def sorted_dict_by_key(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values[key] for key in sorted(values)}

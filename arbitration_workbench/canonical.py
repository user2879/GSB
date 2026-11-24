"""Canonical JSON, hashing and deterministic integer fixed-point helpers."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_text(canonical_json(value))


def parse_decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be numeric")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float, str)):
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
    else:
        raise ValueError(f"{field} must be numeric")
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def fixed_scale(width: int) -> int:
    return 10**width


def to_fixed(value: Decimal, width: int) -> int:
    return int((value * fixed_scale(width)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def fixed_str(value_q: int, width: int) -> str:
    scale = fixed_scale(width)
    sign = "-" if value_q < 0 else ""
    value_q = abs(value_q)
    return f"{sign}{value_q // scale}.{value_q % scale:0{width}d}"


def round_half_up_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    magnitude = abs(numerator)
    return sign * ((2 * magnitude + denominator) // (2 * denominator))


def floor_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator // denominator


def quantize_floor(value_q: int, step: int) -> int:
    if step <= 0:
        raise ValueError("quantization step must be positive")
    return (value_q // step) * step


def int_sqrt_fixed(value_q: int, fractional_bits: int) -> int:
    """sqrt(value / 2**bits) in the same fixed-point scale."""

    if value_q <= 0:
        return 0
    return math.isqrt(value_q << fractional_bits)


def binary_pow2_fraction(numerator: int, denominator: int, fractional_bits: int) -> int:
    """Compute 2**(-numerator/denominator) as an integer scale of 2**bits.

    The fractional exponent is consumed bit by bit.  Successive square roots are
    kept in fixed point and every multiplication is floored.  This is fully
    integer, stable across platforms and does not depend on libm.
    """

    if denominator <= 0 or numerator < 0 or numerator >= denominator:
        raise ValueError("fraction must be in [0, 1)")
    if fractional_bits < 8:
        raise ValueError("fractional_bits too small")
    if numerator == 0:
        return 1 << fractional_bits

    result = 1 << fractional_bits
    root = 1 << (fractional_bits - 1)
    remainder = numerator
    for bit_index in range(1, fractional_bits + 1):
        remainder *= 2
        if remainder >= denominator:
            remainder -= denominator
            result = (result * root) >> fractional_bits
        if remainder == 0:
            break
        if bit_index < fractional_bits:
            root = int_sqrt_fixed(root, fractional_bits)
    return result


def seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy carrying a fingerprint of all content except the fingerprint."""

    content = {key: value for key, value in payload.items() if key != "content_fingerprint"}
    sealed = dict(content)
    sealed["content_fingerprint"] = sha256_obj(content)
    return sealed


def payload_fingerprint_is_valid(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    expected = payload.get("content_fingerprint")
    if not isinstance(expected, str):
        return False
    content = {key: value for key, value in payload.items() if key != "content_fingerprint"}
    return sha256_obj(content) == expected


def chunked(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    chunk: list[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

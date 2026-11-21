from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def number_token(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "nonfinite"
    return format(value, ".17g")


def round_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantize_decimal_to_step(value: Decimal, step: int) -> int:
    if step <= 0:
        raise ValueError("quantization step must be positive")
    units = value / Decimal(step)
    return int(units.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)) * step


def fixed_from_float(value: float, bits: int, tier: int) -> int:
    if not math.isfinite(value):
        raise ValueError("cannot quantize non-finite number")
    scale = 1 << bits
    step = max(1, scale // tier)
    raw = Decimal(str(value)) * Decimal(scale)
    return quantize_decimal_to_step(raw, step)


def fixed_from_decimal(value: Decimal, bits: int, tier: int) -> int:
    if not value.is_finite():
        raise ValueError("cannot quantize non-finite decimal")
    scale = 1 << bits
    step = max(1, scale // tier)
    return quantize_decimal_to_step(value * Decimal(scale), step)


def fixed_to_float(value: int, bits: int) -> float:
    return value / float(1 << bits)


def fixed_mul(left: int, right: int, bits: int) -> int:
    scale = 1 << bits
    return round_div(left * right, scale)


def fixed_div(left: int, right: int, bits: int) -> int:
    scale = 1 << bits
    return round_div(left * scale, right)


def fixed_log2(value: float | Decimal, bits: int, tier: int) -> int:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if decimal <= 0:
        decimal = Decimal(1) / Decimal(1 << bits)
    log_value = Decimal(str(math.log2(float(decimal))))
    return fixed_from_decimal(log_value, bits, tier)


def fixed_sigmoid_margin(margin: int, bits: int) -> int:
    """Map a non-negative margin to [0, 1) in fixed point: m / (m + 1)."""
    scale = 1 << bits
    if margin <= 0:
        return 0
    return fixed_div(margin, margin + scale, bits)


def fixed_to_permille(value: int, bits: int) -> int:
    scale = 1 << bits
    return round_div(value * 1000, scale)


def clamp_fixed(value: int, bits: int) -> int:
    scale = 1 << bits
    return max(0, min(scale, value))

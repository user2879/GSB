"""Small deterministic helpers shared by all stages."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any, *, namespace: str = "") -> str:
    return sha256_text(namespace + "\0" + canonical_json(value))


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decimal_to_q(value: Any, fractional_bits: int) -> int:
    """Convert a YAML number to a deterministic signed fixed-point integer."""

    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    return int((decimal * (1 << fractional_bits)).to_integral_value(rounding="ROUND_HALF_UP"))


def q_to_string(value: int, fractional_bits: int) -> str:
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    scale = 1 << fractional_bits
    digits = min(max(fractional_bits, 1), 18)
    decimal_scale = 10**digits
    scaled = round_div(magnitude * decimal_scale, scale)
    whole, fraction = divmod(scaled, decimal_scale)
    rendered = f"{sign}{whole}.{fraction:0{digits}d}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-"} else "0"


def round_div(numerator: int, denominator: int) -> int:
    """Round half away from zero without using floating point."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        return -((-numerator + denominator // 2) // denominator)
    return (numerator + denominator // 2) // denominator


def clamp(value: int, lower: int, upper: int) -> int:
    return lower if value < lower else upper if value > upper else value


def saturate_signed(value: int, word_bits: int) -> tuple[int, bool]:
    lower = -(1 << (word_bits - 1))
    upper = (1 << (word_bits - 1)) - 1
    if value < lower:
        return lower, True
    if value > upper:
        return upper, True
    return value, False


def derive_bytes(namespace: str, *, block: int = 0) -> bytes:
    return hashlib.sha256(f"{namespace}\0{block}".encode("utf-8")).digest()


def derive_int_vector(
    namespace: str,
    length: int,
    *,
    span: int,
    offset: int = 0,
) -> list[int]:
    """Derive a stable vector in ``[-span, span]`` from SHA-256 words."""

    if span < 0:
        raise ValueError("span must be non-negative")
    width = 2 * span + 1
    result: list[int] = []
    counter = offset
    while len(result) < length:
        digest = derive_bytes(namespace, block=counter)
        for start in range(0, 32, 4):
            word = int.from_bytes(digest[start : start + 4], "big", signed=False)
            result.append((word % width) - span)
            if len(result) == length:
                break
        counter += 1
    return result


def derive_matrix(
    namespace: str,
    rows: int,
    columns: int,
    *,
    span: int,
) -> list[list[int]]:
    values = derive_int_vector(namespace, rows * columns, span=span)
    return [values[row * columns : (row + 1) * columns] for row in range(rows)]


def exp_negative_q(x_q: int, one: int) -> int:
    """Integer Taylor approximation of exp(x), clamped to x <= 0.

    The input and output use the same fixed-point scale. Inputs below -32 are
    saturated to zero. Everything is integer arithmetic and independent of the
    platform's floating-point implementation.
    """

    if x_q >= 0:
        x_q = 0
    if x_q <= -32 * one:
        return 0

    term = one
    total = one
    for index in range(1, 48):
        term = round_div(term * x_q, one * index)
        if term == 0:
            break
        total += term
    return max(0, total)


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))

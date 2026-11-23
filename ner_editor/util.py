"""Stable hashing, JSON normalisation and integer fixed-point helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


def parse_json_text(text: str) -> Any:
    """Parse JSON while preserving decimals (including non-finite values)."""

    return json.loads(
        text,
        parse_float=Decimal,
        parse_int=int,
        parse_constant=lambda value: Decimal(value),
    )


def _number_token(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value.is_nan():
            return {"__decimal__": "NaN"}
        if value.is_infinite():
            return {"__decimal__": "Infinity" if value > 0 else "-Infinity"}
        return {"__decimal__": format(value, "f")}
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "NaN"}
        if math.isinf(value):
            return {"__float__": "Infinity" if value > 0 else "-Infinity"}
        return {"__float__": repr(value)}
    if isinstance(value, Fraction):
        return {"__fraction__": [value.numerator, value.denominator]}
    if isinstance(value, tuple):
        return [_number_token(item) for item in value]
    if isinstance(value, list):
        return [_number_token(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _number_token(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return {"__repr__": repr(value)}


def canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    """Produce stable JSON suitable for content fingerprints and snapshots."""

    return json.dumps(
        _number_token(value),
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_int(parts: str, *, offset: int = 0, size: int = 8) -> int:
    raw = hashlib.sha256(parts.encode("utf-8")).digest()
    chunk = raw[offset : offset + size]
    if len(chunk) < size:
        chunk = (chunk + raw)[:size]
    return int.from_bytes(chunk, "big")


def round_half_away(value: Fraction) -> int:
    """Round an exact fraction to the nearest integer, ties away from zero."""

    if value >= 0:
        shifted = value + Fraction(1, 2)
        return shifted.numerator // shifted.denominator
    return -round_half_away(-value)


def quantize_fraction(value: Fraction, level: int) -> tuple[int, Fraction]:
    """Quantize to a multiple of ``level``; return (units, residual)."""

    if level <= 0:
        raise ValueError("quantization level must be positive")
    quotient = round_half_away(value / level)
    units = quotient * level
    return units, value - units


def quantize_decimal(
    value: Decimal | int | float | str,
    decimals: int,
    level: int,
) -> tuple[int, Fraction]:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError("non-finite value")
    scale = 10**decimals
    return quantize_fraction(Fraction(value) * scale, level)


def fraction_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def residual_direction(residual: Fraction) -> str:
    if residual > 0:
        return "up"
    if residual < 0:
        return "down"
    return "none"


def decimal_from_units(units: int, scale: int) -> Decimal:
    return Decimal(units) / Decimal(scale)


def score_json(units: int, scale: int, decimals: int) -> dict[str, Any]:
    value = decimal_from_units(units, scale)
    return {
        "units": units,
        "scale": scale,
        "fixed_decimals": decimals,
        "decimal": format(value, "f"),
    }


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def stable_sort_dict(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: mapping[key] for key in sorted(mapping)}


def elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def number_to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def value_to_plain(value: Any) -> Any:
    """Convert Decimals to JSON-safe strings only for human-facing extras."""

    if isinstance(value, Decimal):
        if value.is_finite():
            return format(value, "f")
        return str(value)
    if isinstance(value, list):
        return [value_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [value_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): value_to_plain(item) for key, item in value.items()}
    return value

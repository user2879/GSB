"""Canonical JSON, exact rational arithmetic, and digest helpers."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import BusinessError


def load_json_file(path: Path, *, code: str = "E_JSON") -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(code, f"cannot read JSON file: {exc}", str(path)) from exc
    try:
        return json.loads(text, parse_float=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BusinessError(code, f"invalid JSON: {exc}", str(path)) from exc


def load_plain_json_file(path: Path, *, code: str = "E_JSON") -> Any:
    """Load JSON without Decimal parsing, preserving Python's float round-trip."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError(code, f"cannot read JSON file: {exc}", str(path)) from exc
    try:
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BusinessError(code, f"invalid JSON: {exc}", str(path)) from exc


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def as_fraction(value: Any, path: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise BusinessError("E_INPUT", "expected a finite JSON number", path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BusinessError("E_INPUT", "number must be finite", path)
        value = str(value)
    try:
        result = Fraction(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise BusinessError("E_INPUT", "number must be finite", path) from exc
    if isinstance(value, Decimal) and not value.is_finite():
        raise BusinessError("E_INPUT", "number must be finite", path)
    return result


def positive_fraction(value: Any, path: str) -> Fraction:
    result = as_fraction(value, path)
    if result <= 0:
        raise BusinessError("E_INPUT", "number must be strictly positive", path)
    return result


def nonnegative_fraction(value: Any, path: str) -> Fraction:
    result = as_fraction(value, path)
    if result < 0:
        raise BusinessError("E_INPUT", "number must be non-negative", path)
    return result


def as_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BusinessError("E_INPUT", "expected an integer", path)
    if minimum is not None and value < minimum:
        raise BusinessError("E_INPUT", f"integer must be >= {minimum}", path)
    return value


def fraction_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def fraction_from_str(value: str, path: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise BusinessError("E_SNAPSHOT", "invalid rational value in snapshot", path) from exc
    return result


def output_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)


def ceil_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("ceil_fraction expects a non-negative value")
    return (value.numerator + value.denominator - 1) // value.denominator


def download_duration_ms(size_bytes: Fraction, bandwidth_bps: Fraction) -> int:
    if bandwidth_bps <= 0:
        raise ValueError("bandwidth must be positive")
    return ceil_fraction(size_bytes * 8000 / bandwidth_bps)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{id(text)}")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")

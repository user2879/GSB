"""Stable serialization, rational arithmetic and atomic file helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


def as_fraction(value: Any, field: str = "value") -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
        return Fraction(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        try:
            value_fraction = Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{field} must be a finite number") from exc
        return value_fraction
    raise ValueError(f"{field} must be a number")


def round_half_up(value: Fraction) -> int:
    """Round a non-negative rational to the nearest integer, with .5 upward."""
    if value < 0:
        return -round_half_up(-value)
    return (value.numerator * 2 + value.denominator) // (2 * value.denominator)


def quantize(value: Fraction, scale: int) -> int:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return round_half_up(value * scale)


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def format_fixed(value: Fraction | int, scale: int, decimals: int) -> str:
    """Format a fixed-point value without binary floating point."""
    if isinstance(value, int):
        numerator = value
        denominator = scale
    else:
        numerator = round_half_up(value * scale)
        denominator = scale
    if denominator <= 0:
        raise ValueError("invalid fixed-point scale")
    scaled = numerator * (10**decimals)
    quotient, remainder = divmod(scaled, denominator)
    sign = "-" if quotient < 0 else ""
    quotient = abs(quotient)
    integer, fraction = divmod(quotient, 10**decimals)
    if decimals == 0:
        return f"{sign}{integer}"
    return f"{sign}{integer}.{fraction:0{decimals}d}"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(jsonable(value)))


def short_fingerprint(value: Any, length: int = 16) -> str:
    return fingerprint(value)[:length]


def stable_sort_key(*parts: Any) -> str:
    return sha256_text("\x1f".join(str(part) for part in parts))


def json_fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def jsonable(value: Any) -> Any:
    """Convert model values into JSON-safe deterministic primitives."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Fraction):
        return json_fraction(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value, key=repr)]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    _fsync_directory(path)


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value, ensure_ascii=False))
    return result


def ordered_unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
        try:
            already = marker in seen
        except TypeError:
            already = False
        if not already:
            seen.add(marker)
            result.append(value)
    return result

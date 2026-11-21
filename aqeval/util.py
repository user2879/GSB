"""Canonical serialization, exact rounding and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any


MICRO_SCALE = 1_000_000
PERMILLE_SCALE = 1_000


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_half_up_fraction(value: Fraction) -> int:
    """Round a Fraction to the nearest integer, with ties upward."""

    if value < 0:
        return -round_half_up_fraction(-value)
    return (value.numerator * 2 + value.denominator) // (value.denominator * 2)


def round_fraction_to_scale(value: Fraction, scale: int) -> int:
    return round_half_up_fraction(value * scale)


def decimal_to_micro(value: Decimal | float | int) -> int:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return round_half_up_fraction(Fraction(value * MICRO_SCALE))


def micro_to_float(value: int) -> float:
    return value / MICRO_SCALE


def micro_to_permille(value: int) -> int:
    return (value + PERMILLE_SCALE // 2) // PERMILLE_SCALE


def decimal_fraction(value: Decimal | float | int | str) -> Fraction:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return Fraction(value)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, pretty_json_bytes(value))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonable(value: Any) -> Any:
    """Convert Decimal and tuple containers into canonical JSON values."""

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def clamp(value: int | float, low: int | float, high: int | float) -> int | float:
    return max(low, min(high, value))

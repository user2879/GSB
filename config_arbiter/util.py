from __future__ import annotations

import hashlib
import json
import os
import tempfile
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any


def canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_text(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def fraction_json(value: Fraction) -> dict[str, Any]:
    numerator = value.numerator
    denominator = value.denominator
    decimal = Decimal(numerator) / Decimal(denominator)
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "decimal": text or "0",
    }


def parse_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return parsed


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

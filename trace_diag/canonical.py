from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def fraction_json(value: Fraction | int) -> dict[str, int]:
    item = Fraction(value)
    return {"numerator": item.numerator, "denominator": item.denominator}


def as_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, dict) and "numerator" in value and "denominator" in value:
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    raise TypeError(f"not a rational value: {value!r}")


def fraction_decimal(value: Fraction | int, places: int = 3) -> str:
    item = Fraction(value)
    scale = 10**places
    scaled = item.numerator * scale // item.denominator
    sign = "-" if scaled < 0 else ""
    scaled = abs(scaled)
    whole, remainder = divmod(scaled, scale)
    return f"{sign}{whole}.{remainder:0{places}d}"

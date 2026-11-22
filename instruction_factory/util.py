from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

from .constants import SAFE_ID_PATTERN


class CanonicalJSONError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def parse_json_text(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError(str(exc)) from exc


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_text(text: str) -> str:
    return fingerprint_bytes(text.encode("utf-8"))


def fingerprint_object(value: Any) -> str:
    return fingerprint_text(canonical_json(value))


def fingerprint_file(path: str | Path) -> str:
    return fingerprint_bytes(Path(path).read_bytes())


def stable_int(*parts: Any, bits: int = 128) -> int:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def stable_key_hash(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(SAFE_ID_PATTERN, value) is not None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    if not is_number(value):
        return False
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return False
    return True


def as_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean is not a decimal number")
    return Decimal(str(value))


def decimal_text(value: Decimal, places: int = 12) -> str:
    if value == 0:
        return "0"
    quantum = Decimal(1).scaleb(-places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def rounded_float(value: Decimal | float, places: int = 10) -> float:
    quantum = Decimal(1).scaleb(-places)
    if not isinstance(value, Decimal):
        value = as_decimal(value)
    return float(value.quantize(quantum, rounding=ROUND_HALF_UP))


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def read_json_file(path: str | Path) -> Any:
    return parse_json_text(Path(path).read_text(encoding="utf-8"))


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


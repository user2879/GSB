from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def stable_sorted(values: Iterable[Any], key=None) -> list[Any]:
    return sorted(values, key=key or (lambda item: item))


def failure(stage: str, code: str, stable_key: str, message: str, **details: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "stage": stage,
        "code": code,
        "stable_key": stable_key,
        "message": message,
    }
    item.update(details)
    return item


def failure_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("stage", "")),
        str(item.get("stable_key", "")),
        str(item.get("code", "")),
        str(item.get("message", "")),
    )


def sort_failures(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=failure_sort_key)


def first_difference(left: Any, right: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            key = sorted(left_keys ^ right_keys)[0]
            return f"{path}.{key}", left.get(key, "<missing>"), right.get(key, "<missing>")
        for key in sorted(left_keys):
            result = first_difference(left[key], right[key], f"{path}.{key}")
            if result:
                return result
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result = first_difference(left_item, right_item, f"{path}[{index}]")
            if result:
                return result
        return None
    if left != right:
        return path, left, right
    return None


def round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        return -round_div(-numerator, denominator)
    return (numerator + denominator // 2) // denominator


def fixed_point_score(raw_millionths: int, fixed_decimal_bits: int, quantum: int) -> int:
    if fixed_decimal_bits < 0 or fixed_decimal_bits > 60:
        raise ValueError("fixed_decimal_bits out of range")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    scale = 1 << fixed_decimal_bits
    numerator = raw_millionths * scale
    denominator = 1_000_000 * quantum
    return round_div(numerator, denominator) * quantum


def format_fixed(score_fixed: int, fixed_decimal_bits: int) -> str:
    if fixed_decimal_bits == 0:
        return str(score_fixed)
    scale = 1 << fixed_decimal_bits
    sign = "-" if score_fixed < 0 else ""
    value = abs(score_fixed)
    whole, remainder = divmod(value, scale)
    digits = max(1, math.ceil(fixed_decimal_bits * math.log10(2)))
    fraction = (remainder * (10**digits)) // scale
    return f"{sign}{whole}.{fraction:0{digits}d}"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]

"""Stable serialization, hashing and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_text(value: Any, *, newline: bool = True) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return text + ("\n" if newline else "")


def stable_hash(value: Any, *, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return digest if length is None else digest[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(file_sha256(child).encode("ascii") + b"\n")
    return digest.hexdigest()


def clean_float(value: float, significant_digits: int | None = None) -> float:
    """Return a finite float, normalizing negative zero and optional sig digits."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number: {number!r}")
    if number == 0.0:
        return 0.0
    if significant_digits is not None:
        if significant_digits < 1:
            raise ValueError("significant_digits must be positive")
        number = float(format(number, f".{significant_digits}g"))
        if number == 0.0:
            return 0.0
    return number


def round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json_text(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(flatten_numeric(item))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("weight elements must be numeric")
    return [float(value)]


def chunk_ranges(start: int, end: int, limit: int) -> list[tuple[int, int]]:
    if limit <= 0:
        raise ValueError("chunk limit must be positive")
    return [(cursor, min(cursor + limit, end)) for cursor in range(start, end, limit)]


def first_present(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default

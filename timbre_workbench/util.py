"""Canonical hashing, stable rounding and atomic file helpers."""

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


def sha256_obj(value: Any) -> str:
    return sha256_text(canonical_json(value))


def cache_key(stage: str, *parts: Any, key_version: int = 1) -> str:
    payload = {"stage": stage, "key_version": key_version, "parts": list(parts)}
    return sha256_obj(payload)


def stable_float(value: float, digits: int = 9) -> float:
    if not math.isfinite(value):
        return 0.0
    rounded = round(float(value), digits)
    if abs(rounded) < 0.5 * 10 ** (-digits):
        return 0.0
    return rounded


def round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def round_div_half_away(numerator: int, denominator: int) -> int:
    """Divide integers with deterministic round-half-away-from-zero."""
    if denominator == 0:
        raise ZeroDivisionError("denominator must be non-zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    n = abs(int(numerator))
    d = abs(int(denominator))
    return sign * ((n + d // 2) // d)


def normalize_relpath(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def safe_join(root: Path, relative: str) -> Path:
    relative_path = Path(normalize_relpath(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe relative path: {relative!r}")
    return root.joinpath(relative_path)


def sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted(set(values), key=lambda value: (value.casefold(), value))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


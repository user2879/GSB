"""Canonical hashing helpers shared by every pipeline stage."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def hash_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def digest_short(value: str, length: int = 16) -> str:
    return value[:length]


def finite_float(value: float, *, digits: int = 15) -> float:
    """Round a float for stable JSON output and reject non-finite values."""

    if not math.isfinite(value):
        raise ValueError("non-finite value")
    return float(f"{value:.{digits}g}")

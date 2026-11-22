"""Stable content fingerprints used by cache, snapshots and reports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


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


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_token(value: float | int | None) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, bool):
        raise TypeError("booleans are not numeric feature values")
    if isinstance(value, int):
        return f"int:{value}"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite feature value")
    return f"float:{number.hex()}"

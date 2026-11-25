"""Canonical serialization and fingerprint helpers.

All persisted fingerprints use sorted-key, compact JSON with UTF-8 encoding.
This makes hashes stable across process restarts and thread scheduling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def stable_number(value: Any) -> Any:
    """Normalize integral floats so hashes and evidence do not drift by spelling."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

"""Canonical hashing helpers used by cache keys and snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_float(value: float) -> str:
    """Serialize a finite float in a platform-stable, lossless-ish form."""

    if not math.isfinite(value):
        raise ValueError("non-finite float cannot be canonicalized")
    return float(value).hex()

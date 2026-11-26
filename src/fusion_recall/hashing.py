from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic canonical JSON bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def short_hash(value: Any, length: int = 16) -> str:
    return canonical_hash(value)[:length]


def composite_key(prefix: str, *parts: Any) -> str:
    material = {
        "prefix": prefix,
        "parts": parts,
    }
    return f"{prefix}_{canonical_hash(material)[:24]}"

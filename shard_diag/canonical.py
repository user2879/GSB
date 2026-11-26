from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Return the repository-wide canonical JSON byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_digest(value: Any, prefix: str = "sha256") -> str:
    return f"{prefix}:{sha256_hex(canonical_bytes(value))}"


def stable_uint32(material: str) -> int:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 0xFFFFFFFF


def saturate_int64(value: int) -> int:
    """Saturate to signed int64. Inputs are arbitrary Python integers."""

    lo = -(2**63)
    hi = 2**63 - 1
    if value > hi:
        return hi
    if value < lo:
        return lo
    return int(value)

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Return the explicit byte serialization used by all identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return digest if length is None else digest[:length]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def monotonic_ms() -> int:
    import time

    return time.monotonic_ns() // 1_000_000

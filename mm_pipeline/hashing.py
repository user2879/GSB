"""Content-addressed fingerprinting.

All fingerprints are SHA-256 over a canonical byte representation, so a hash is
a pure function of (input content, parameters, quantization level). Nothing
here reads the wall clock, process id or filesystem ordering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON encoding: sorted keys, fixed separators, no float surprises."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_bytes(data: bytes, *, prefix: str = "") -> str:
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode("utf-8"))
        h.update(b"\x1f")
    h.update(data)
    return h.hexdigest()


def hash_text(text: str, *, prefix: str = "") -> str:
    return hash_bytes(text.encode("utf-8"), prefix=prefix)


def hash_json(obj: Any, *, prefix: str = "") -> str:
    return hash_bytes(canonical_bytes(obj), prefix=prefix)


def combine(*parts: Any, prefix: str = "") -> str:
    """Combine an ordered set of parts into one stable fingerprint."""
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode("utf-8"))
        h.update(b"\x1f")
    for part in parts:
        if isinstance(part, bytes):
            chunk = part
        elif isinstance(part, str):
            chunk = part.encode("utf-8")
        else:
            chunk = canonical_bytes(part)
        h.update(str(len(chunk)).encode("ascii"))
        h.update(b":")
        h.update(chunk)
        h.update(b"|")
    return h.hexdigest()


def cache_key(prefix: str, *parts: Any) -> str:
    """Namespaced cache key for a stage artifact."""
    return combine(*parts, prefix=prefix)


def short(key: str, length: int = 16) -> str:
    return key[:length]


def hash_iterable(values: Iterable[Any], *, prefix: str = "") -> str:
    """Fingerprint an ordered sequence without materializing a giant JSON blob."""
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode("utf-8"))
        h.update(b"\x1f")
    for value in values:
        chunk = canonical_bytes(value) if not isinstance(value, (str, bytes)) else (
            value.encode("utf-8") if isinstance(value, str) else value
        )
        h.update(chunk)
        h.update(b"|")
    return h.hexdigest()

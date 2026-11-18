"""Deterministic serialization and domain-separated hashing helpers.

Every hash used by the workbench is SHA-256 over UTF-8 bytes produced by
:func:`canonical_bytes`.  Hash combination always uses explicit length and
domain prefixes so that distinct structures can never collide into the same
digest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

FRAME_OPEN = b"\x01"
FRAME_CLOSE = b"\x02"


def canonical_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` with sorted keys and stable separators, as UTF-8."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def feed_domained(hasher, domain: str, payload: bytes) -> None:
    """Append a framed ``(domain, payload)`` block to a running hash."""
    hasher.update(FRAME_OPEN)
    hasher.update(domain.encode("utf-8"))
    hasher.update(b":")
    hasher.update(str(len(payload)).encode("ascii"))
    hasher.update(FRAME_CLOSE)
    hasher.update(payload)


def domained_hash(domain: str, obj: Any) -> str:
    hasher = hashlib.sha256()
    feed_domained(hasher, domain, canonical_bytes(obj))
    return hasher.hexdigest()


def combined_hash(domain: str, parts: Iterable[tuple[str, Any]]) -> str:
    """Hash a stable, ordered sequence of named parts."""
    hasher = hashlib.sha256()
    feed_domained(hasher, "root:" + domain, b"v1")
    for name, value in parts:
        feed_domained(hasher, name, canonical_bytes(value))
    return hasher.hexdigest()


def chained_line_hash(domain: str, lines: Iterable[bytes]) -> str:
    """Hash an ordered stream of pre-serialized lines (streaming-safe)."""
    hasher = hashlib.sha256()
    feed_domained(hasher, "stream:" + domain, b"v1")
    for index, line in enumerate(lines):
        framed = str(index).encode("ascii") + b":" + line
        feed_domained(hasher, "row", framed)
    feed_domained(hasher, "eof", b"")
    return hasher.hexdigest()


def short_hexdigest(digest: str) -> str:
    return digest[:16]

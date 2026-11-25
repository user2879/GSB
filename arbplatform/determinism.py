"""Deterministic serialization, content digests and monotonic timing.

Arbitration results must be byte-identical regardless of dictionary / hash
iteration order, report ordering or batch splitting. All persisted and
compared structures therefore flow through :func:`canonical_json`, which
sorts mapping keys, emits integers without float artifacts and uses a fixed
separator set.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def canonical(obj: Any) -> str:
    """Return a canonical JSON text with sorted keys and compact separators."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_of(obj: Any) -> str:
    """SHA-256 hex digest of the canonical encoding of ``obj``."""
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def now_ms() -> int:
    """Monotonic clock elapsed milliseconds (integer, non-decreasing)."""
    return int(time.monotonic() * 1000)


def sorted_items(mapping: Any) -> list[tuple[str, Any]]:
    """Return mapping items sorted by key for deterministic traversal."""
    return sorted(mapping.items(), key=lambda kv: kv[0])


def stable_dumps_line(obj: Any) -> bytes:
    """Encode one NDJSON line deterministically (newline terminated)."""
    return (canonical(obj) + "\n").encode("utf-8")

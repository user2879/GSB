"""Deterministic canonical JSON encoding and content digests.

Every persisted conclusion and snapshot uses :func:`canonical_digest` so that
the same logical input always produces the same bytes, regardless of dict
insertion order, set ordering or process id.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    """Encode *obj* as canonical JSON bytes.

    Rules: object keys sorted lexicographically, no insignificant whitespace,
    ``sort_keys`` applied recursively, and non-finite floats rejected so that
    output is always reproducible and transport-safe.
    """

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(obj: Any) -> str:
    """Return the SHA-256 hex digest of the canonical encoding of *obj*."""

    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def canonical_json(obj: Any, indent: int | None = None) -> str:
    if indent is None:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    return json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=False,
                      allow_nan=False)

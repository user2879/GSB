"""Deterministic canonical JSON and content-digest helpers.

All byte-identity comparisons (window results, snapshot digests, replay) go
through :func:`canonical_bytes` so that dict ordering, whitespace and tuple
encoding can never introduce spurious differences.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    """Return a stable UTF-8 byte representation of ``obj``.

    Dictionaries are emitted with sorted keys, no whitespace is emitted and
    tuples are treated exactly like lists. Only JSON-native scalar types and
    the containers dict/list/tuple are supported.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(obj: Any) -> str:
    """SHA-256 hex digest of the canonical representation of ``obj``."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()

"""Deterministic, byte-stable canonical encoding.

All "byte-identical" guarantees in the platform (compiled artifacts, view rows,
authoritative replay, snapshot digests) go through :func:`canonical_bytes`.

Rules:
* dict keys are sorted recursively;
* no insignificant whitespace is emitted;
* non-finite floats are rejected (all data is finite);
* the encoding is UTF-8 with ``ensure_ascii=False`` and a stable key order.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _check_finite(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be canonically encoded")
    elif isinstance(value, dict):
        for sub in value.values():
            _check_finite(sub)
    elif isinstance(value, (list, tuple)):
        for sub in value:
            _check_finite(sub)


def canonical_bytes(value: Any) -> bytes:
    """Return the platform-wide canonical byte representation of ``value``."""
    _check_finite(value)
    text = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def digest(value: Any) -> str:
    """SHA-256 hex digest of the canonical encoding."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

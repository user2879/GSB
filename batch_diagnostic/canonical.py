"""Explicit, deterministic serialization primitives.

All identifiers and content digests in the component are derived from
``canonical_json`` output: keys are sorted, no insignificant whitespace is
emitted, and only JSON-native scalar types are accepted. The resulting bytes are
therefore independent of dict insertion order, dispatch order, or platform.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_ALLOWED = (str, int, float, bool, type(None), list, tuple, dict)


def canonical_json(obj: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for ``obj``.

    Tuples are treated as lists. Integer keys are converted to strings. Any
    non-JSON-native type raises ``TypeError`` so accidental nondeterminism (for
    example sets) cannot leak into identifiers.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): normalize(value[key]) for key in sorted(value, key=str)}
        if isinstance(value, _ALLOWED):  # pragma: no cover - defensive
            return value
        raise TypeError(f"cannot canonically serialize type {type(value).__name__}")

    return json.dumps(
        normalize(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    """Stable SHA-256 hex digest over canonical JSON bytes."""

    return hashlib.sha256(canonical_json(obj)).hexdigest()


def short_digest(obj: Any, length: int = 16) -> str:
    """Truncated digest used inside human-readable identifiers."""

    return sha256_hex(obj)[:length]

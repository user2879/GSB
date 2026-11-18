"""Content fingerprint / cache-key helpers (SHA-256 based, deterministic)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_json(obj: Any) -> str:
    """Canonical JSON text used for fingerprinting structured values."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    """Stable fingerprint for any JSON-serialisable value."""
    return sha256_text(canonical_json(obj))


def cache_key(prefix: str, parts: Iterable[Any]) -> str:
    """Compose a stage cache key from labelled fingerprint parts.

    Each part is first rendered through :func:`canonical_json` so strings,
    numbers, lists and nested structures all contribute deterministically.
    """
    joined = "|".join(canonical_json(part) for part in parts)
    return f"{prefix}:{sha256_text(joined)}"


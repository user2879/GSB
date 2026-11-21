"""Content fingerprints used for cache keys and snapshot manifests."""

from __future__ import annotations

import hashlib
import json


def canonical(obj) -> str:
    """Canonical JSON serialization: sorted keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(obj) -> str:
    """SHA-256 content fingerprint of any JSON-serialisable object."""
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def short(fp: str, n: int = 12) -> str:
    return fp[:n]

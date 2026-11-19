"""Content fingerprints and stage cache keys.

All hashing uses BLAKE2b with a fixed digest size so keys are stable across
runs, Python versions and process restarts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DIGEST_SIZE = 16


def stable_dumps(obj: Any) -> str:
    """Deterministic JSON encoding (sorted keys, no whitespace surprises)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj: Any) -> str:
    """Return a stable hex fingerprint for any JSON-serialisable object."""
    payload = stable_dumps(obj)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=DIGEST_SIZE).hexdigest()


def fingerprint_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=DIGEST_SIZE).hexdigest()


def combine(*parts: str) -> str:
    """Combine several fingerprints/parameters into one derived key."""
    joined = "|".join(parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=DIGEST_SIZE).hexdigest()


def stage_key(stage: str, *parts: str) -> str:
    """Build a cache key for ``stage`` from content/parameter fingerprints."""
    return combine(stage, *parts)

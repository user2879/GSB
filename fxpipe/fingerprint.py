"""Stable content fingerprints derived from canonical JSON payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj: Any, prefix: str = "fp") -> str:
    digest = hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:16]}"


def cache_key(parts: list[Any], prefix: str = "key") -> str:
    """Cache key = hash of upstream fingerprints, op params, gear, version."""
    return fingerprint(parts, prefix=prefix)


def config_hash(config_dict: dict) -> str:
    return fingerprint(config_dict, prefix="cfg")

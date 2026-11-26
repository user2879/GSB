"""Content / parameter fingerprinting.

Fingerprints are BLAKE2b digests over a canonical JSON encoding.  Only data and
explicit version markers enter the hash; wall-clock time, thread ids and
filesystem ordering never do, so fingerprints are identical across runs,
machines and concurrency levels.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

DIGEST_LEN = 16


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (set, tuple)):
        return sorted(obj) if obj and not isinstance(next(iter(obj)), dict) else list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"object not canonicalizable: {type(obj)!r}")


def fingerprint(obj: Any, prefix: str = "") -> str:
    payload = (prefix + "|" + canonical(obj)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=DIGEST_LEN).hexdigest()


def combine(parts: list[str], prefix: str = "combine") -> str:
    return fingerprint(parts, prefix)


def key_for(version: int, stage: str, parts: list[str]) -> str:
    """Cache key = snapshot version + stage + upstream fingerprints."""
    return f"v{version}:{stage}:" + combine([str(version), stage] + parts, stage)

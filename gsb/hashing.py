from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize data in the stable form used by fingerprints and snapshots."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def short_fp(value: Any, length: int = 12) -> str:
    if not isinstance(value, str):
        value = fingerprint(value)
    return value[:length]


def cache_key(stage: str, *parts: Any) -> str:
    material = "|".join(str(part) for part in parts)
    return f"{stage}:{sha256_text(material)[:20]}"

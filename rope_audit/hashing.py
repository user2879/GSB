from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any, length: int = 16) -> str:
    raw = canonical_json(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def prefixed_key(prefix: str, *parts: Any) -> str:
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"

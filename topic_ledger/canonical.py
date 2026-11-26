from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize data in the single canonical form used by fingerprints."""
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


def short_fp(value: Any, length: int = 16) -> str:
    return fingerprint(value)[:length]


def stable_sorted(values: list[Any]) -> list[Any]:
    return sorted(values, key=lambda item: canonical_json(item))

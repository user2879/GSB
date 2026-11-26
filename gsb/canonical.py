from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Serialize JSON-compatible data in one canonical byte representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def short_fingerprint(value: Any, length: int = 16) -> str:
    return fingerprint(value)[:length]


def combine_fingerprints(*values: Any) -> str:
    return fingerprint(list(values))

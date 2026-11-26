"""Canonical JSON and digest helpers.

The canonical form is UTF-8 JSON with sorted object keys, compact separators
and no non-ASCII escaping.  It is deliberately independent of dictionary and
hash iteration order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))

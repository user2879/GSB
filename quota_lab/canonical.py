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


def sha256_hex(value: bytes | str | Any) -> str:
    if not isinstance(value, bytes):
        value = canonical_bytes(value)
    return hashlib.sha256(value).hexdigest()


def short_digest(value: Any, length: int = 12) -> str:
    return sha256_hex(value)[:length]

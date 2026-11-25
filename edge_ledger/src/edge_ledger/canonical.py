from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Stable JSON used by digests and replay; never relies on insertion order."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_default,
    )


def _default(value: Any) -> Any:
    if hasattr(value, "to_canonical"):
        return value.to_canonical()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_hex(canonical_dumps(value))

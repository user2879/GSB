from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Serialize data in a stable, byte-exact JSON form."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_hex(value: Any | bytes) -> str:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    else:
        data = canonical_bytes(value)
    return hashlib.sha256(data).hexdigest()


def content_digest(record: dict[str, Any]) -> str:
    """Digest a record without its own digest field."""

    core = {key: item for key, item in record.items() if key != "content_digest"}
    return sha256_hex(core)

"""Canonical stable serialization and content digests.

The canonical JSON form is the single byte-identical representation used for
batch identifiers, ledger digests and adoption conclusions: sorted keys,
compact separators no whitespace, and ``ensure_ascii=False``.
"""

import hashlib
import json
from typing import Any

CANONICAL_KWARGS = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
}


def canonical_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return json.dumps(value, **CANONICAL_KWARGS).encode("utf-8")


def canonical_text(value: Any) -> str:
    return json.dumps(value, **CANONICAL_KWARGS)


def sha256_hex(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def short_id(value: Any, length: int = 32) -> str:
    return sha256_hex(value)[:length]

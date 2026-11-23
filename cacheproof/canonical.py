"""Canonical JSON and hashing helpers.

All persisted JSON that participates in a digest is encoded by these helpers:
UTF-8, sorted object keys, no insignificant whitespace, no trailing newline.
Arrays retain caller-provided order. Business code always sorts arrays before
calling these helpers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_value(value: Any) -> str:
    return "sha256:" + sha256_hex(canonical_bytes(value))


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")



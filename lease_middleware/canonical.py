"""Canonical JSON and digest helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return (canonical_text(value) + "\n").encode("utf-8")


def sha256_hex(value: bytes | str | Any) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        value = canonical_bytes(value)
    return hashlib.sha256(value).hexdigest()


def content_digest(value: Any) -> str:
    return "sha256:" + sha256_hex(canonical_bytes(value))

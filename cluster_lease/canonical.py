"""Canonical serialization and digest helpers.

All persisted and protocol-visible content uses these helpers.  Digests never
depend on Python dictionary insertion order or hash randomization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
PROTOCOL_VERSION = 1
WATERMARK_SCALE = 1_000_000
WATERMARK_TRUNCATION = "floor"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def watermark_fixed(used_capacity: int, capacity: int) -> int:
    """Return used/capacity scaled by WATERMARK_SCALE, truncated down."""

    if capacity <= 0:
        return 0
    return (used_capacity * WATERMARK_SCALE) // capacity

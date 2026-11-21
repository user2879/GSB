"""Canonical JSON and digest helpers.

All persisted records use this module so that the same semantic value has one
byte representation across processes, restarts and ledger verification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def body_digest(body: Any) -> str:
    return sha256_hex(canonical_bytes(body))


def ledger_digest(payload: Any) -> str:
    return sha256_hex(canonical_bytes(payload))


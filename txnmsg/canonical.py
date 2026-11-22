"""Canonical JSON and digest helpers.

All persisted digests cover UTF-8 canonical JSON (sorted keys, compact
separators).  This makes conclusions independent of dictionary insertion
order, JSON formatting and batch boundaries.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


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


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_value(value: Any) -> str:
    return digest_bytes(canonical_bytes(value))


def message_envelope(partition_id: str, position: int, payload: Any) -> dict[str, Any]:
    return {
        "partition_id": partition_id,
        "position": position,
        "payload": payload,
    }


def message_digest(partition_id: str, position: int, payload: Any) -> str:
    return digest_value(message_envelope(partition_id, position, payload))

"""Canonical serialization and deterministic digest helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a value with one stable representation across runs."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_hex(*parts: Any) -> str:
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        if index:
            digest.update(b"\x1f")
        if not isinstance(part, (bytes, bytearray)):
            part = canonical_json(part).encode("utf-8")
        digest.update(part)
    return digest.hexdigest()


def short_fingerprint(*parts: Any, length: int = 16) -> str:
    return sha256_hex(*parts)[:length]


def stable_task_key(combo_fingerprint: str, stage: str) -> str:
    """Stable sort key carried by every stage task."""

    return sha256_hex("task-key-v1", stage, combo_fingerprint)


def stable_salt(combo_fingerprint: str) -> bytes:
    return hashlib.sha256(
        b"category-cross-salt-v1\x1f" + combo_fingerprint.encode("ascii")
    ).digest()


def hash_cross_value(salt: bytes, canonical_value: str, encoding_width: int) -> int:
    """Hash a canonical tuple and reduce it to the configured bit width."""

    digest = hashlib.blake2b(
        canonical_value.encode("utf-8"), digest_size=32, salt=salt[:16]
    ).digest()
    value = int.from_bytes(digest, "big", signed=False)
    if encoding_width >= 256:
        return value
    return value & ((1 << encoding_width) - 1)

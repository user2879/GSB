"""Deterministic canonical JSON hashing.

``canonical_bytes`` is the single byte-level representation used by both the
SQLite ledger and the snapshot machinery, so an independent recompute can
compare results byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def digest_of(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def chain_digest(previous: str, payload: Any) -> str:
    merged = {"prev": previous, "payload": payload}
    return digest_of(merged)

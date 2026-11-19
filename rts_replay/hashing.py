"""Deterministic canonical-JSON serialization and content hashing.

All persisted artifacts are serialized with :func:`canonical_dumps`, which sorts
mapping keys and emits compact UTF-8 JSON. Feeding the exact same data structure
always yields byte-identical output on every platform and Python version, which
is the basis of the byte-for-byte reproducibility requirement.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_dumps(obj: Any) -> bytes:
    """Serialize *obj* to canonical JSON bytes (sorted keys, compact separators)."""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(obj: Any) -> str:
    """Return the SHA-256 hex digest of the canonical serialization of *obj*."""
    return hashlib.sha256(canonical_dumps(obj)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_digest(digest: str) -> str:
    return digest[:12]

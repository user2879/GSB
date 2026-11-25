"""Deterministic canonical serialization and content hashing.

Two Python objects that are semantically equal always serialize to the exact
same byte string.  Mapping keys are sorted lexicographically by their UTF-8
encoding, separators are fixed, no insignificant whitespace is emitted and
integers never gain a decimal point.  These byte strings are the single
source of truth used for batch ids, internal representations, ledger content
summaries and snapshot integrity.
"""
from __future__ import annotations

import hashlib
from typing import Any

_CANONICAL_SEPARATORS = (",", ":")


def canonical_bytes(obj: Any) -> bytes:
    """Return the canonical UTF-8 byte representation of ``obj``."""
    import json

    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=_CANONICAL_SEPARATORS,
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(obj: Any) -> str:
    return canonical_bytes(obj).decode("utf-8")


def content_hash(obj: Any) -> str:
    """SHA-256 hex digest of the canonical representation."""
    material = obj if isinstance(obj, (bytes, bytearray)) else canonical_bytes(obj)
    return hashlib.sha256(bytes(material)).hexdigest()


def short_hash(obj: Any, length: int = 16) -> str:
    return content_hash(obj)[:length]

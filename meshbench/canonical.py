"""Byte-stable canonical JSON encoding and content digests.

Recomputed receipts are compared *byte for byte* with cached receipts, so every
component serializes through the same deterministic encoder: sorted keys, no
insignificant whitespace, and separators fixed once for the whole session.
"""

import hashlib
import json

CANONICAL_SEPARATORS = (",", ":")


def canonical_dumps(value) -> bytes:
    """Return canonical UTF-8 JSON bytes, independent of insertion order."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=CANONICAL_SEPARATORS,
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value) -> str:
    """Hex BLAKE2b digest of the canonical encoding of ``value``."""
    return hashlib.blake2b(canonical_dumps(value), digest_size=32).hexdigest()


def digest_bytes(data: bytes) -> str:
    """Hex BLAKE2b digest over raw bytes."""
    return hashlib.blake2b(data, digest_size=32).hexdigest()

"""Canonical byte encoding and content digests.

Every persisted payload and every stream summary goes through these helpers so
that equal data always produces equal bytes, independent of dict insertion or
hash traversal order.
"""

import hashlib
import json


def canonical_json(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

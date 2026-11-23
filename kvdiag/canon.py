"""Canonical serialization and stable digests.

All persisted/comparable artifacts go through canonical_json so results are
byte-identical regardless of dict insertion or hash iteration order.
"""

import hashlib
import json


def canonical_json(obj):
    """Serialize deterministically: sorted keys, tight separators, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_digest(obj):
    """SHA-256 hex digest of the canonical serialization of obj."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def key_digest(key):
    """Stable digest for an integer key (truncated SHA-256 hex)."""
    return hashlib.sha256(("key:%d" % key).encode("utf-8")).hexdigest()[:16]

"""Canonical serialization and digests used across the pipeline."""

import hashlib
import json


def canonical_bytes(obj):
    """Deterministic JSON encoding (sorted keys, tight separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def digest_obj(obj):
    return sha256_hex(canonical_bytes(obj))

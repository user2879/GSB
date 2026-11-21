"""Stable hashing helpers: canonical JSON, content digests and gray buckets.

All hashing uses SHA-256 over canonical JSON (sort_keys, no whitespace) so that
the same logical object always yields the same digest on every platform and
Python version. Floating point values are never placed into digests directly;
they are rendered through deterministic helpers in :mod:`hotgray.determinism`.
"""

import hashlib
import json

CANONICAL_KW = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, **CANONICAL_KW).encode("utf-8")


def sha256_hex(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_digest(obj) -> str:
    """Digest of the logical content of ``obj`` (with any digest field dropped)."""
    if isinstance(obj, dict) and "digest" in obj:
        obj = {k: v for k, v in obj.items() if k != "digest"}
    return sha256_hex(canonical_bytes(obj))


def stable_bucket(rule_id: str, client_key: str) -> int:
    """Map (rule_id, client_key) deterministically into [0, 9999].

    Gray weight is expressed in basis points (per ten thousand). A request goes
    to the gray backend when ``bucket < weight``; the same inputs therefore
    always produce the same decision for the same weight.
    """
    digest = sha256_hex(canonical_bytes([rule_id, client_key]))
    return int(digest[:8], 16) % 10000


def route_decision(rule_id: str, client_key: str, weight_bp: int) -> str:
    return "gray" if stable_bucket(rule_id, client_key) < weight_bp else "stable"

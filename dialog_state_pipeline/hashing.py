"""Deterministic fingerprinting helpers.

Everything funnels through :func:`canonical_json` which sorts dictionary keys
and uses explicit separators, so fingerprints are independent of dict order
and the platform C hash seed.  SHA-256 is truncated to 16 bytes which is
ample for local cache keys while keeping manifest lines short.
"""

import hashlib
import json


def canonical_json(obj):
    """Return a stable JSON string for *obj* (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj):
    """Stable SHA-256 hex digest of any JSON-serialisable object."""
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_digest(obj):
    """64-character SHA-256 prefix (128 bits) used for cache keys."""
    return digest(obj)[:32]


def chain_digest(prev_key, payload):
    """Hash-chain step: combine the previous turn key with a payload.

The chain makes every downstream cache key depend on *all* upstream turns,
which is what gives precise yet safe pollution propagation: editing one turn
changes its own key and therefore every key after it.
    """
    material = prev_key + "|" + canonical_json(payload)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


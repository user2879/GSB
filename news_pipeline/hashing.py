"""Canonical JSON serialization and content fingerprinting.

Cache keys are sha256 digests over canonical JSON.  Dict keys are sorted and
tuples/lists keep their declared order, so the fingerprint is independent of
insertion order or the way a container was built.
"""

import hashlib
import json


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(obj):
    payload = canonical(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combine(*parts):
    """Fingerprint a sequence of already-stable parts (strings or objects)."""
    normalized = [p if isinstance(p, str) else canonical(p) for p in parts]
    return fingerprint(normalized)

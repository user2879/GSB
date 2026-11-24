"""Deterministic canonical serialization and SHA-256 fingerprint helpers.

No object is ever hashed via Python's ``hash()``/``id()``; every key is built
from an explicit, order-preserving canonical JSON document so that identical
inputs always map to the same key across processes and platforms.
"""

import hashlib
import json

import numpy as np


def canonical_bytes(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=_default,
    ).encode("utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"non-canonical object: {type(o)!r}")


def stable_hash(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def short_hash(obj, length=16):
    return stable_hash(obj)[:length]


def array_fingerprint(array, decimals=10):
    """Fingerprint of a numeric array independent of memory layout."""
    arr = np.asarray(array, dtype=np.float64)
    rounded = np.round(arr, decimals)
    return stable_hash(["ndarray", str(arr.shape), rounded.tolist()])


def combine_keys(*parts):
    return stable_hash(["combine", list(parts)])

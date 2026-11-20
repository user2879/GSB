"""Content hashing primitives.

All hashes are SHA-256 over a canonical byte representation so that two
identical logical objects always produce byte-identical digests, independent
of dict insertion order or platform pointer values.
"""

import hashlib
import json

import numpy as np


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_array(arr):
    """Hash a numpy array over its raw little-endian machine bytes.

    The dtype and shape are folded in so identical buffers of different
    shapes/dtypes cannot collide.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("hash_array expects an np.ndarray")
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return h.hexdigest()


def canonical_json(obj):
    """Deterministic JSON text: sorted keys, no platform float surprises."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_json(obj):
    return hash_bytes(canonical_json(obj).encode("utf-8"))

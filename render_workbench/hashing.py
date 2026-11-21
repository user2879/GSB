"""Deterministic hashing, fingerprinting and RNG primitives.

No global random state is used anywhere in the workbench. Every random source
is a fresh ``numpy.random.RandomState`` seeded from explicit inputs.
"""

import hashlib
import json

import numpy as np


def canonical_bytes(obj):
    """Serialize ``obj`` to canonical, key-order-independent JSON bytes."""
    return json.dumps(
        _jsonable(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist()}
    return obj


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint(obj):
    """Content fingerprint for JSON-compatible structures."""
    return "sha256:" + sha256_hex(canonical_bytes(obj))


def array_fingerprint(arr):
    """Content fingerprint for a numpy array based on its exact bytes."""
    arr = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|")
    digest.update(arr.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def bytes_fingerprint(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def derive_seed(*parts):
    """Derive a 32-bit integer seed from arbitrary JSON-compatible parts."""
    joined = b"|".join(canonical_bytes(p) for p in parts)
    digest = hashlib.sha256(b"seed-v1|" + joined).digest()
    return int.from_bytes(digest[:4], "big")


def random_state(*parts):
    """Return an isolated RandomState derived from explicit inputs."""
    return np.random.RandomState(derive_seed(*parts))


def cache_key(kind, params, upstream_fingerprints):
    """Deterministic cache key for a pipeline node.

    ``params`` captures the node parameters, ``upstream_fingerprints`` the
    content fingerprints of every upstream artifact.
    """
    material = {
        "kind": kind,
        "params": _jsonable(params),
        "upstream": [str(fp) for fp in upstream_fingerprints],
    }
    return "cache:" + kind + ":" + sha256_hex(canonical_bytes(material))[:32]

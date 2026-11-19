"""Deterministic hashing / canonical serialisation helpers."""
import hashlib
import json

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    """Canonical JSON: sorted keys, tight separators, no timestamps allowed in
    the payload by convention (callers must keep committed content
    deterministic)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def array_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return sha256_bytes(a.tobytes())


def floats_hex(values) -> list:
    """Exact, platform-stable float serialisation for result hashing."""
    return [float(v).hex() for v in np.asarray(values, dtype=np.float64).ravel()]

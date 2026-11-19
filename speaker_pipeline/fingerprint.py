"""Deterministic content and parameter fingerprinting.

All fingerprints are SHA-256 hex digests computed over canonical JSON or
raw bytes, so identical inputs always produce identical keys across runs,
processes and thread schedules.
"""

import hashlib
import json

import numpy as np


def canonical_bytes(obj):
    """Serialize *obj* to stable bytes (sorted keys, no whitespace tricks)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_fingerprint(obj):
    """Fingerprint for arbitrary JSON-serializable content."""
    return sha256_hex(canonical_bytes(obj))


def param_fingerprint(params):
    """Fingerprint for a parameter mapping (order independent)."""
    return sha256_hex(canonical_bytes(params))


def bytes_fingerprint(data):
    """Fingerprint for raw byte content (e.g. WAV file payload)."""
    return sha256_hex(data)


def derived_key(stage, *upstream_keys, **params):
    """Derive a cache key from upstream product keys and parameters."""
    payload = {
        "stage": stage,
        "upstream": list(upstream_keys),
        "params": params,
    }
    return stage + ":" + sha256_hex(canonical_bytes(payload))[:16]


def vector_fingerprint(vector):
    """Deterministic fingerprint for a 1-D float64 vector."""
    arr = np.asarray(vector, dtype=np.float64)
    return sha256_hex(arr.tobytes())


def round6(value):
    """Round to 6 decimal places, keeping at least 6 significant digits
    for the magnitudes used here and making serialization stable."""
    return round(float(value), 6)

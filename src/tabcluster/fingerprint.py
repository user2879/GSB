"""Canonical serialization and content fingerprints.

Every artifact carries a content fingerprint derived from upstream content
fingerprints and parameter fingerprints, so cache invalidation is exact.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _normalize(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": bytes(obj).hex()}
    if isinstance(obj, dict):
        return {str(k): _normalize(obj[k]) for k in obj}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        # JSON round-trips IEEE754 doubles exactly (shortest repr); NaN is
        # stored as null so artifacts stay valid JSON and compare equal.
        if obj != obj:
            return None
        return obj
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def fingerprint(*parts: Any) -> str:
    """sha256 over canonical serialization of all parts, order-sensitive."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            blob = part.encode("utf-8")
        else:
            blob = canonical_json(part).encode("utf-8")
        h.update(len(blob).to_bytes(8, "big"))
        h.update(blob)
    return h.hexdigest()


def array_fingerprint(arr: np.ndarray) -> str:
    """Fingerprint of a float64 ndarray: shape + raw bytes (C order)."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    h = hashlib.sha256()
    h.update(repr(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def short(fp_value: str, n: int = 16) -> str:
    return fp_value[:n]

"""Content/parameter fingerprints used as cache keys and lineage edges."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _feed(parts: list[bytes]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(str(len(part)).encode("ascii"))
        h.update(b":")
        h.update(part)
    return h.hexdigest()


def fingerprint_bytes(data: bytes, namespace: str) -> str:
    return _feed([namespace.encode("utf-8"), data])


def fingerprint_array(arr: np.ndarray, namespace: str) -> str:
    return _feed([namespace.encode("utf-8"), np.ascontiguousarray(arr).tobytes(),
                  str(arr.shape).encode("utf-8"), str(arr.dtype).encode("utf-8")])


def fingerprint_params(namespace: str, params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _feed([namespace.encode("utf-8"), payload])


def derive_cache_key(namespace: str, upstream: list[str], params: dict[str, Any]) -> str:
    parts = [namespace.encode("utf-8")]
    for name in sorted(upstream):
        parts.append(name.encode("utf-8"))
    parts.append(json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return _feed(parts)

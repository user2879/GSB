"""Content and parameter fingerprints used for cache keys and invalidation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint64(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big", signed=False)


def array_fingerprint(values: np.ndarray) -> int:
    arr = np.ascontiguousarray(values)
    header = f"{arr.dtype.str}:{arr.shape}:{arr.strides}".encode("ascii")
    h = hashlib.sha256()
    h.update(header)
    h.update(b"\0")
    h.update(arr.tobytes())
    return fingerprint64(h.digest())


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def parameter_fingerprint(obj: Any) -> int:
    return fingerprint64(canonical_json_bytes(obj))


def combine_fingerprints(*parts: Any) -> int:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            part = array_fingerprint(part)
        h.update(str(part).encode("ascii"))
        h.update(b"|")
    return fingerprint64(h.digest())


def config_hash(config: dict[str, Any], materials: list[dict[str, Any]]) -> str:
    """Stable hash covering generation config and material descriptions."""
    payload = canonical_json_bytes({"config": config, "materials": materials})
    return sha256_hex(payload)


def cache_key(kind: str, *parts: Any) -> str:
    return f"{kind}:{combine_fingerprints(kind, *parts):016x}"

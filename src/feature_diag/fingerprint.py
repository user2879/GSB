"""Stable content fingerprints and pseudo-random streams.

The workbench never relies on process-global RNG state.  Every random-looking
value is derived from SHA-256 over canonical text, which makes generated data,
hash buckets, and random projections stable across runs and thread schedules.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def fingerprint(value: Any) -> str:
    return sha256_text(canonical_json(value))


def array_fingerprint(array: np.ndarray) -> str:
    arr = np.asarray(array)
    return fingerprint(
        {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tolist(),
        }
    )


def cache_key(
    *,
    kind: str,
    upstream_fingerprint: str,
    parameter_fingerprint: str,
    precision: str,
    schema_version: int = 1,
) -> str:
    """Derive a cache key exactly from upstream, parameters, and precision."""

    return fingerprint(
        {
            "schema": schema_version,
            "kind": kind,
            "upstream": upstream_fingerprint,
            "parameters": parameter_fingerprint,
            "precision": precision,
        }
    )


def stable_uint64(*parts: Any, counter: int = 0) -> int:
    payload = "|".join(str(part) for part in parts) + f"|c={counter}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def stable_uniform(*parts: Any, counter: int = 0) -> float:
    return (stable_uint64(*parts, counter=counter) + 0.5) / float(2**64)


def stable_normal(*parts: Any, counter: int = 0) -> float:
    u1 = stable_uniform(*parts, counter=counter * 2)
    u2 = stable_uniform(*parts, counter=counter * 2 + 1)
    radius = math.sqrt(-2.0 * math.log(max(u1, 1e-300)))
    return radius * math.cos(2.0 * math.pi * u2)


def stable_seed(*parts: Any) -> int:
    return stable_uint64(*parts) & 0xFFFFFFFF


def fingerprint_parts(*parts: Iterable[Any]) -> str:
    return fingerprint([list(part) if not isinstance(part, str) else part for part in parts])

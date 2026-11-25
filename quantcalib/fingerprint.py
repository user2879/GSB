"""Deterministic content fingerprints.

All fingerprints are SHA-256 hex digests over canonical byte encodings so that
two runs on different machines / thread schedules produce identical keys.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np


def _feed(hasher: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, np.ndarray):
        hasher.update(b"ndarray")
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(value.shape.__repr__().encode("utf-8"))
        hasher.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        hasher.update(b"{")
        for key in sorted(value):
            hasher.update(repr(str(key)).encode("utf-8"))
            _feed(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, (list, tuple)):
        hasher.update(b"[")
        for item in value:
            _feed(hasher, item)
        hasher.update(b"]")
    else:
        hasher.update(repr(value).encode("utf-8"))


def fingerprint(value: Any) -> str:
    hasher = hashlib.sha256()
    _feed(hasher, value)
    return hasher.hexdigest()


def fingerprint_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_bytes(parts: Iterable[bytes]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part)
    return hashlib.hexdigest()


def short(fp: str, length: int = 12) -> str:
    return fp[:length]

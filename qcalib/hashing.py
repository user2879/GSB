"""Stable content/parameter fingerprints used as cache key material."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def fingerprint(payload: Any, prefix: str = "") -> str:
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:16]}" if prefix else digest[:16]


def array_fingerprint(values: np.ndarray, prefix: str = "w-") -> str:
    flat = np.ascontiguousarray(values, dtype=np.float64).ravel()
    digest = hashlib.sha256(flat.tobytes()).hexdigest()
    return f"{prefix}{digest[:16]}"


def composite_key(parts: dict[str, str]) -> str:
    return fingerprint(parts, prefix="key-")

"""Content fingerprinting primitives.

Fingerprints are SHA-256 hex digests computed from canonical byte streams.
Nothing here depends on process-wide random state, object identity or wall
clock time, so identical logical inputs always yield identical fingerprints on
every machine and every run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value into a canonical byte string.

    Keys are sorted and separators are fixed, so dict ordering at the call site
    never affects the result.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_json(value: Any) -> str:
    return sha256_hex(canonical_bytes(value))


def fingerprint_array(array: np.ndarray) -> str:
    """Fingerprint a numpy array from shape, dtype and its raw byte content."""
    arr = np.ascontiguousarray(array)
    payload = canonical_bytes(
        {
            "kind": "ndarray",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data_b64_sha": sha256_hex(arr.tobytes(order="C")),
        }
    )
    return sha256_hex(payload)


def array_summary(array: np.ndarray) -> dict:
    """Deterministic, finite-safe numeric summary for audit reports."""
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(arr.size - finite.sum()),
        "content_fingerprint": fingerprint_array(arr),
    }
    if finite.any():
        flat = arr[finite].astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(flat.min()),
                "max": float(flat.max()),
                "mean": float(flat.mean()),
                "sum": float(flat.sum()),
            }
        )
    return summary


def fingerprint_pixels(pixels: np.ndarray) -> str:
    return fingerprint_array(pixels)

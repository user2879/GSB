"""Deterministic fingerprints used by the column-level cache."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def column_content_fingerprint(
    values: np.ndarray,
    missing_mask: np.ndarray,
    original_dtype: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"column-content-v1\0")
    digest.update(original_dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(missing_mask, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def stage_cache_key(stage: str, *inputs: str, params: Any) -> str:
    payload = {
        "stage": stage,
        "inputs": list(inputs),
        "params_fingerprint": sha256_json(params),
    }
    return sha256_json(payload)

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def array_digest(array: np.ndarray) -> str:
    array64 = np.ascontiguousarray(array, dtype="<f8")
    return sha256_bytes(
        canonical_json_bytes(
            {"dtype": "<f8", "shape": list(array64.shape), "order": "C"}
        ),
        array64.tobytes(order="C"),
    )


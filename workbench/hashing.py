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


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_hex(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def array_fingerprint(array: np.ndarray) -> str:
    """Hash dtype, shape and C-order pixels, not object identity/strides."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def short_version(fingerprint: str, length: int = 16) -> str:
    return fingerprint[:length]


def stable_task_key(stage: str, level: int, row: int, col: int) -> str:
    """Stable lexicographic key. Zero padding preserves numeric ordering."""
    return f"{stage}:L{level:02d}:R{row:04d}:C{col:04d}"


def stable_key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

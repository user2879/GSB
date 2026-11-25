from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any, length: int = 20) -> str:
    return sha256_hex(canonical_json(value))[:length]


def full_fingerprint(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def array_fingerprint(array: np.ndarray | list[Any]) -> str:
    arr = np.ascontiguousarray(array)
    header = canonical_json({"shape": list(arr.shape), "dtype": arr.dtype.str})
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def cache_key(*parts: Any) -> str:
    return fingerprint({"parts": parts}, length=32)

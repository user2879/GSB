from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_fingerprint(value),
        }
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def sha256_hex(data: bytes | str) -> str:
    digest = hashlib.sha256()
    if isinstance(data, str):
        data = data.encode("utf-8")
    digest.update(data)
    return digest.hexdigest()


def fingerprint_json(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def derive_seed(*parts: Any) -> int:
    return int(fingerprint_json(list(parts))[:16], 16) & 0x7FFFFFFFFFFFFFFF


def array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_hex(
        canonical_json([str(contiguous.dtype), list(contiguous.shape)]).encode()
        + contiguous.tobytes(order="C")
    )


def file_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(*parts: Any) -> str:
    return fingerprint_json({"cache_key_version": 1, "parts": parts})

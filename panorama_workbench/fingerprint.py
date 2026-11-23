from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def hash_bytes(*parts: bytes | str | int) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            encoded = part
        else:
            encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_content_hash(rgb: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)
    return hash_bytes(
        f"rgb:{contiguous.shape[1]}x{contiguous.shape[0]}:{contiguous.shape[2]}".encode(
            "ascii"
        ),
        contiguous.tobytes(),
    )


def tile_content_hash(tile: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(tile, dtype=np.uint8)
    return hash_bytes(
        f"tile:{contiguous.shape[1]}x{contiguous.shape[0]}".encode("ascii"),
        contiguous.tobytes(),
    )


def array_hash(array: np.ndarray, dtype: str | None = None) -> str:
    target = np.ascontiguousarray(array, dtype=dtype) if dtype else np.ascontiguousarray(array)
    return hash_bytes(
        f"array:{target.dtype.str}:{target.shape}".encode("ascii"), target.tobytes()
    )

"""规范化 JSON 哈希与张量内容指纹。"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def canonical_json(obj) -> str:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(obj) -> str:
    return sha256_text(canonical_json(obj))


def fingerprint_array(a: np.ndarray) -> str:
    """张量内容指纹：dtype/形状 + 小端原始字节。"""
    arr = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(d) for d in arr.shape).encode("utf-8"))
    h.update(b"|")
    h.update(arr.astype(arr.dtype, copy=False).tobytes(order="C"))
    return h.hexdigest()


def fingerprint_int(a: np.ndarray) -> str:
    return fingerprint_array(a)

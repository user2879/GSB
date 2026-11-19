"""确定性哈希工具：对字节、numpy 数组与参数对象计算稳定 sha256。"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """按内存中的实际 dtype/形状计算哈希，浮点也逐位可复现。"""
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("ascii"))
    h.update(b"|")
    h.update(",".join(str(d) for d in contiguous.shape).encode("ascii"))
    h.update(b"|")
    h.update(contiguous.tobytes(order="C"))
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """排序键、固定分隔符的规范 JSON，用于参数指纹。"""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def short(h: str, n: int = 12) -> str:
    return h[:n]

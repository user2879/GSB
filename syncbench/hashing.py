"""内容哈希与规范化序列化。

- JSON 以 sort_keys + 固定分隔符 + 不允许 NaN 规范化，保证跨平台字节一致；
- 节点哈希对“规范化 JSON 载荷”计算 SHA-256；二进制块单独哈希后入载荷；
- numpy 数组以小端原始字节哈希（dtype/形状纳入盐值）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np


def canonical_bytes(obj: Any) -> bytes:
    """将可 JSON 化对象编码为确定性字节（禁止 NaN/Infinity）。"""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(obj: Any) -> str:
    return canonical_bytes(obj).decode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))


def hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(b"ARRAY\1")
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(str(a.shape).encode("ascii"))
    h.update(b"\0")
    h.update(a.astype(a.dtype, copy=False).tobytes(order="C"))
    return h.hexdigest()


def hash_blobs(hashes: Iterable[str]) -> str:
    h = hashlib.sha256()
    h.update(b"BLOBS\1")
    for item in sorted(hashes):
        h.update(item.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def payload_hash(payload: Any, blob_hashes: Iterable[str] | None = None) -> str:
    """节点内容哈希 = 哈希(规范化载荷 + 关联二进制块哈希列表)。"""
    material = {"payload": payload}
    if blob_hashes is not None:
        material["blobs"] = sorted(blob_hashes)
    return hash_json(material)

"""确定性内容指纹工具。

全部使用纯整数运算与 SHA-256，保证同一输入多次运行逐字节一致，
且不依赖 Python 对象哈希的随机化（PYTHONHASHSEED）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence


def stable_dumps(obj) -> str:
    """以排序键、无空白差异的形式序列化 JSON。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(obj) -> str:
    """任意 JSON 可序列化对象的内容指纹（16 位短哈希）。"""
    return sha256_hex(stable_dumps(obj))[:16]


def ints_fingerprint(values: Sequence[int] | Iterable[int]) -> str:
    """整数序列指纹，逐整数编码，避免分隔符歧义。"""
    digest = hashlib.sha256()
    count = 0
    for value in values:
        digest.update(int(value).to_bytes(8, byteorder="big", signed=True))
        count += 1
    digest.update(count.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()[:16]


def combine_fingerprints(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()[:16]

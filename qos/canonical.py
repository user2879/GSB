"""确定性序列化与哈希工具。

规则要求编译产物的字节表示不依赖 dict/set 的遍历顺序，
因此统一使用 sort_keys=True、固定分隔符与 LF 换行的 JSON 编码。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    """返回确定性、可逐字节比较的 JSON 字节串。"""
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(obj: Any) -> str:
    return canonical_bytes(obj).decode("utf-8")


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def short_hash(obj: Any) -> str:
    return sha256_hex(obj)[:16]

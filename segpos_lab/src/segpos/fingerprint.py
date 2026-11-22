"""确定性指纹工具。

所有指纹均通过规范化 JSON（排序键、无空白、ASCII 安全）后取 SHA-256，
保证同输入逐字节复现，且不依赖 Python 对象的内存哈希随机化。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(obj: Any, prefix: str = "") -> str:
    payload = canonical(obj)
    if prefix:
        payload = prefix + "|" + payload
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def combine(prefix: str, *parts: Any) -> str:
    """多源指纹：先规范化各部分再拼接，避免分隔歧义。"""
    joined = [prefix]
    for part in parts:
        if isinstance(part, str) and len(part) == 64:
            joined.append(part)
        else:
            joined.append(fingerprint(part))
    return hashlib.sha256("|".join(joined).encode("utf-8")).hexdigest()

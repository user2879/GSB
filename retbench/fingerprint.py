"""确定性内容/参数指纹工具。

全部指纹使用 BLAKE2b（Python 标准库，长度 16 字节 hex），对嵌套结构先做
规范化 JSON 序列化（sort_keys、紧凑分隔符、保证 ASCII），再哈希。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_DIGEST_LEN = 16


def canonical_json(obj: Any) -> str:
    """规范化 JSON：键排序、无空白，保证同构对象字符串一致。"""

    return json.dumps(
        obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=_default
    )


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"无法对类型 {type(obj)!r} 计算指纹")


def digest_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=_DIGEST_LEN).hexdigest()


def digest_text(text: str) -> str:
    return digest_bytes(text.encode("utf-8"))


def digest_obj(obj: Any) -> str:
    return digest_text(canonical_json(obj))


def digest_many(*parts: Any) -> str:
    """组合多个上游指纹/标量，生成派生指纹。"""

    joined = "|".join(
        p if isinstance(p, str) else canonical_json(p) for p in parts
    )
    return digest_text(joined)

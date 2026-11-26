"""确定性序列化与内容摘要工具。

所有摘要均基于键排序的 JSON 规范形式（canonical form），保证跨进程、
跨重开与跨平台逐字节可复核，不依赖字典/集合遍历顺序。
"""

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    """返回带键排序、紧凑分隔的规范 JSON 文本。"""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def digest(obj: Any) -> str:
    """对任意可 JSON 化对象计算 sha256 摘要（十六进制）。"""
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def digest_text(text: str) -> str:
    """对已确定的文本计算 sha256 摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

"""确定性序列化与内容摘要。

所有摘要均基于显式排序规则后的规范 JSON，不依赖字典或哈希遍历顺序，
保证同一图内容在任意构建顺序下得到逐字节一致的摘要。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    """生成规范 JSON 文本：键排序、无空白、UTF-8 不转义。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj: Any) -> str:
    """对任意可 JSON 序列化对象计算确定性内容摘要。"""
    return "sha256:" + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()

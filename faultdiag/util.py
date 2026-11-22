"""通用确定性工具: 单调时钟毫秒、规范化 JSON 与内容摘要。"""

from __future__ import annotations

import hashlib
import json
import time


def monotonic_ms() -> int:
    """单调时钟毫秒, 用于迁移与排队等待的时间记录。"""
    return time.monotonic_ns() // 1_000_000


def canonical(obj) -> str:
    """规范化 JSON 序列化(键排序、无空白), 保证逐字节稳定。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_obj(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

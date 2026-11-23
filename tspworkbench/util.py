"""确定性序列化与内容指纹工具。

指纹一律走 ``json.dumps(sort_keys=True, separators=(",", ":"))`` 的规范字节，
保证不同进程、不同字典插入顺序下逐字节稳定，从而让增量与全量键一致。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    """生成跨运行稳定的规范 JSON 字符串。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(obj: Any) -> str:
    """对任意可 JSON 化对象计算 SHA-256 内容指纹（16 位短哈希）。"""
    digest = hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()
    return digest[:16]


def derive_key(parts: list[Any]) -> str:
    """由多个上游内容/参数/版本指纹派生缓存键。"""
    return fingerprint([fingerprint(p) if not isinstance(p, str) else p
                        for p in parts])


def stable_sort_key(series_id: str, window_index: int, stage: str) -> str:
    """窗口任务的稳定排序键：序列 ID、窗口序号、阶段名派生。"""
    return f"{series_id}|{window_index:08d}|{stage}"

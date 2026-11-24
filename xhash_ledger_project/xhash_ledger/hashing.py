"""内容指纹与确定性派生工具。

全部基于 SHA-256，输入先做规范化 JSON（键排序、无空白、保留分隔符），
保证同内容在任何机器/运行次数下指纹一致，且不依赖对象内存地址或 dict 迭代顺序。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_json(obj: Any) -> str:
    """规范化 JSON 文本：键排序、无多余空白，保证跨平台字节一致。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint(obj: Any) -> str:
    """对任意可 JSON 化对象计算内容指纹（SHA-256 十六进制）。"""
    return sha256_hex(canonical_json(obj))


def derive_key(parts: Iterable[Any]) -> str:
    """把多个上游指纹/版本字段组合为一个缓存键。"""
    material = [p if isinstance(p, str) else canonical_json(p) for p in parts]
    return sha256_hex("|".join(material))


def stable_int(seed_text: str) -> int:
    """由任意文本派生一个稳定的 64 位非负整数，用于投影矩阵种子。"""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def int_to_bucket_key(value: int, bucket_bits: int) -> str:
    """把签名子串整数格式化为定长十六进制桶键。"""
    width = max(1, (bucket_bits + 3) // 4)
    return f"{value:0{width}x}"

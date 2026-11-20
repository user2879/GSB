"""跨进程、跨线程、跨字典遍历顺序稳定的确定性工具。

禁止使用 Python 内置 ``hash()`` 处理路由（其随机化种子逐进程变化），
统一使用 FNV-1a 64 位；摘要使用 SHA-256；JSON 使用排序键 + 固定分隔符。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

FNV_OFFSET_64 = 0xCBF29CE484222325
FNV_PRIME_64 = 0x100000001B3
FNV_MASK_64 = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(data: bytes) -> int:
    """返回 FNV-1a 64 位哈希值（无符号整数）。"""
    value = FNV_OFFSET_64
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME_64) & FNV_MASK_64
    return value


def stable_hash(text: str) -> int:
    """对字符串做稳定哈希，等价于对 UTF-8 字节做 FNV-1a。"""
    return fnv1a_64(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """排序键、无多余空白的规范 JSON 文本（ensure_ascii=False）。"""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def content_digest(obj: Any) -> str:
    """对任意可 JSON 化对象计算内容摘要（SHA-256 十六进制）。"""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def stable_targets(targets: Iterable[str]) -> list[str]:
    """把物理库表标识集合规范为排序后的列表，保证集合语义序列化稳定。"""
    return sorted(set(targets))


def physical_target(db_index: int, table_index: int) -> str:
    """统一的物理库表标识格式：``db{库号}.t{表号}``。"""
    return f"db{db_index}.t{table_index}"


def all_physical_targets(db_count: int, tables_per_db: int) -> list[str]:
    """枚举全部物理库表（广播目标），按库号、表号稳定排序。"""
    return [
        physical_target(db_index, table_index)
        for db_index in range(db_count)
        for table_index in range(tables_per_db)
    ]


def split_target(target: str) -> tuple[int, int]:
    """解析 :func:`physical_target` 产生的标识，返回 (库号, 表号)。"""
    db_part, table_part = target.split(".")
    return int(db_part[2:]), int(table_part[1:])

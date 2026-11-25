"""与字典/哈希遍历顺序无关的规范化序列化与内容摘要。

核心约定：
- 任何需要逐字节复核的结构都先经 :func:`canonical_bytes` 规范化；
- 字典按键的 Unicode 码点序排序，列表保持声明的业务序；
- 不允许集合、元组等顺序不稳定类型直接进入规范化；
- 浮点使用最短往返 repr，并把 -0.0 归一为 0.0，避免符号位噪声。
"""

import hashlib
import json


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(value[key]) for key in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, set):
        raise TypeError("set 不能参与规范化序列化，需先转为排序列表")
    return value


def canonical_bytes(value) -> bytes:
    """返回确定性的 UTF-8 JSON 字节串。"""
    text = json.dumps(
        _normalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_text(value) -> str:
    return canonical_bytes(value).decode("utf-8")


def digest(value) -> str:
    """对规范化字节计算 SHA-256，返回 64 位小写十六进制摘要。"""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def short_digest(value, length: int = 16) -> str:
    return digest(value)[:length]

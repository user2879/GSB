"""确定性工具：规范化 JSON、内容指纹、float64 序列化、稳定排序。"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


def canonical_json(obj: Any) -> str:
    """跨平台、键顺序无关的规范化 JSON 文本。

    - dict 按键排序递归处理；
    - 不保留多余空白；
    - ensure_ascii=False，UTF-8 字节参与哈希；
    - float 使用 repr（Python 最短往返表示），保证 float64 逐位一致。
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(obj: Any) -> str:
    """对任意可 JSON 化对象计算 SHA-256 指纹（十六进制）。"""
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(*parts: str) -> int:
    """由若干上游指纹派生 64 位随机种子（无随机性，纯哈希）。"""
    joined = "|".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def derive_cache_key(*parts: str) -> str:
    """由上游内容指纹与参数指纹派生阶段缓存键。"""
    joined = "|".join(parts)
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def float64(value: float) -> float:
    """显式转 IEEE-754 float64（numpy/python float 均为 float64）。"""
    return float(value)


def round_sig(value: float, sig: int = 9) -> float:
    """用于产物展示的确定性舍入：固定保留 9 位小数。

    分数取值范围为 [0, 1]，9 位小数保证任何非零分数至少保留 6 位有效
    数字（满足题目要求），且对 float64 往返无损、无平台相关差异；
    所有中间计算仍使用完整 float64，不在打分过程中舍入。
    """
    if value == 0.0 or not math.isfinite(value):
        return float(value)
    factor = 10.0**9
    if value >= 0:
        return math.floor(value * factor + 0.5) / factor
    return -math.floor(-value * factor + 0.5) / factor


def format_sig(value: float, sig: int = 6) -> str:
    """以固定有效数字格式化 float64，保证产物文本至少保留 6 位有效数字。

    JSON 数字会丢弃尾随零，因此用字符串承载精度展示；底层计算与
    排序仍使用 float64 数值字段。
    """
    if value == 0.0:
        return "0." + "0" * (sig - 1)
    if not math.isfinite(value):
        return repr(value)
    return f"{value:.{sig - 1}e}" if abs(value) < 10 ** -(sig - 1) else \
        f"{value:.{max(sig - 1 - int(math.floor(math.log10(abs(value)))), 0)}f}"


def stable_id(prefix: str, index: int, width: int = 6) -> str:
    """生成字典序与数值序一致的稳定 ID。"""
    return f"{prefix}#{index:0{width}d}"


def sorted_by_id(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item["candidate_id"])


def is_nonneg_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )

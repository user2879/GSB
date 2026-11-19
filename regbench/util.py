"""确定性工具：稳定 JSON、内容哈希、种子派生、单调时钟封装。"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


def stable_json(obj: Any) -> str:
    """按键排序、无空白差异的 JSON 序列化，用于指纹与缓存键。"""

    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_default
    )


def _default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"无法稳定序列化对象: {type(obj)!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    return sha256_text(stable_json(obj))


def derive_seed(*parts: Any) -> int:
    """由任意可序列化片段确定性派生 [0, 2^32) 内的 NumPy 种子。"""

    digest = sha256_text(stable_json(list(parts)))
    return int(digest[:8], 16)


def monotonic() -> float:
    """单调时钟读数（秒）。仅用于报告展示，不参与任何指纹/一致性比较。"""

    return time.monotonic()


def truncate_list(values: list[float], head: int = 5, tail: int = 5) -> dict[str, Any]:
    """逐轮损失轨迹摘要：首/尾若干点与长度，避免报告膨胀。"""

    length = len(values)
    if length <= head + tail:
        return {"length": length, "head": list(values), "tail": []}
    return {
        "length": length,
        "head": list(values[:head]),
        "tail": list(values[-tail:]),
    }

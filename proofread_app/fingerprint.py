"""内容指纹与派生缓存键。

全部指纹使用 SHA-256，序列化采用 ``sort_keys=True`` 的紧凑 JSON，
保证不同进程 / 不同线程调度顺序下指纹完全一致，不依赖对象内存地址。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Tuple

_FP_SENTINEL = "\u0000fp\u0000"


def canonical_json(obj: Any) -> str:
    """稳定 JSON 序列化：键排序、紧凑分隔、不转义非 ASCII。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "to_fingerprint_obj"):
        return obj.to_fingerprint_obj()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"不可指纹化的对象类型: {type(obj)!r}")


def content_fingerprint(obj: Any) -> str:
    """计算任意可 JSON 化对象的内容指纹（16 位短哈希，碰撞概率可忽略）。"""
    payload = canonical_json(obj)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def derive_key(upstream: Iterable[str], *parts: Any) -> str:
    """由上游指纹与本阶段参数指纹派生缓存键。

    ``upstream`` 为上游阶段产物指纹（已排序），``parts`` 为本阶段配置/参数。
    任一上游内容或本阶段参数变化都会改变缓存键。
    """
    upstream_tuple: Tuple[str, ...] = tuple(sorted(str(u) for u in upstream))
    material = canonical_json({"upstream": upstream_tuple, "params": list(parts)})
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return digest[:20]


def stable_order_key(sample_id: str, start: int, stage: str) -> str:
    """任务稳定排序键：样本 ID、起始位置、阶段名。"""
    return f"{sample_id}\u0001{start:08d}\u0001{stage}"


def combine_fingerprints(*fps: str) -> str:
    return content_fingerprint([_FP_SENTINEL, *fps])

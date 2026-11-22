"""规范化序列化与内容哈希。

所有跨进程/跨次运行需要逐字段比对的产物，都经 canonical_json 落盘，
键排序、紧凑分隔、固定浮点表示，保证字节级确定。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def to_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(to_native(obj), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(obj))
        fh.write("\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.loads(fh.read())

"""确定性内容指纹与缓存键工具。

全部哈希基于纯 Python 标准库 ``hashlib``（sha256），
不依赖进程级 ``hash()`` 随机化，保证跨进程、跨机器可复现。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def canonical_bytes(value: Any) -> bytes:
    """把任意 JSON 兼容结构/NumPy 数组序列化为确定性字节串。"""
    return json.dumps(
        _to_native(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _to_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.astype(np.float64).tobytes().hex(),
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    return value


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value: Any) -> str:
    """计算任意可序列化值（含 ndarray）的内容指纹。"""
    return sha256_hex(canonical_bytes(value))


def fingerprint_arrays(arrays: Iterable[np.ndarray]) -> str:
    parts = [
        {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.astype(np.float64).tobytes().hex(),
        }
        for array in arrays
    ]
    return fingerprint(parts)


def fingerprint_bytes(raw: bytes) -> str:
    return sha256_hex(raw)


def cache_key(stage: str, param_fingerprint: str, deps: Mapping[str, str]) -> str:
    """由阶段名、参数指纹与依赖指纹清单派生稳定缓存键。"""
    payload = {
        "stage": stage,
        "params": param_fingerprint,
        "deps": {name: deps[name] for name in sorted(deps)},
    }
    return sha256_hex(canonical_bytes(payload))


def stable_fingerprint_of_keys(keys: Sequence[str]) -> str:
    return fingerprint(sorted(keys))

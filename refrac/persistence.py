"""产物持久化与内容哈希。

- float32 RGBA 缓冲以 numpy 原始字节落盘（.npy），哈希按原始字节计算；
- JSON 产物使用规范序列化（sort_keys + 固定缩进），保证逐字节可复现。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np

from .scene import dumps_canonical


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_rgba(path: str, rgba: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, rgba)  # 写入 <path>.npy
    actual_path = path if path.endswith(".npy") else path + ".npy"
    with open(actual_path, "rb") as handle:
        return sha256_bytes(handle.read())


def save_json(path: str, payload: Any) -> tuple[str, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = dumps_canonical(payload) if not isinstance(payload, str) else payload
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path, sha256_bytes(text.encode("utf-8"))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

"""确定性摘要工具。

摘要基于数组的原始字节计算（配合固定 dtype 与 C 序内存布局），
因此逐位相同的数组会得到相同摘要，任何单字节差异都会改变摘要。
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def array_digest(arr: np.ndarray) -> str:
    """返回数组原始字节的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def token_digest(tokens: Iterable[str]) -> str:
    """把若干字符串令牌按顺序合成为一个稳定摘要。"""
    h = hashlib.sha256()
    for token in tokens:
        h.update(token.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def first_diff_index(a: np.ndarray, b: np.ndarray) -> int | None:
    """返回两个同形数组首个不一致的扁平元素序号；完全一致返回 None。"""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch in first_diff_index")
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    eq = flat_a == flat_b
    if np.all(eq):
        return None
    return int(np.flatnonzero(~eq)[0])


def bytewise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """逐字节比较两个数组（dtype 与形状也必须一致）。"""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()

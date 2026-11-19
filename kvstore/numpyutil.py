"""NumPy 向量化校验工具。

把“前缀计数视图 vs 按键集合重算”以及“可见计数 vs 逻辑时钟过滤”建模为整数
矩阵比较，借助 NumPy 一次性找出全部偏差并返回首个偏差前缀，供 E_INDEX 报告。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from . import codec
from .errors import IndexError as KvIndexError


def compare_count_vectors(
    cached: dict[str, int], recomputed: dict[str, int]
) -> dict[str, Any]:
    prefixes = sorted(set(cached) | set(recomputed))
    cached_vec = np.array([cached.get(p, 0) for p in prefixes], dtype=np.int64)
    expected_vec = np.array([recomputed.get(p, 0) for p in prefixes], dtype=np.int64)
    mismatch = np.flatnonzero(cached_vec != expected_vec)
    if mismatch.size:
        index = int(mismatch[0])
        prefix = prefixes[index]
        raise KvIndexError(
            "NumPy 向量比对发现计数偏差",
            details={
                "prefix": prefix,
                "expected_from_keyset": int(expected_vec[index]),
                "actual_cached": int(cached_vec[index]),
                "total_prefixes": len(prefixes),
                "mismatch_count": int(mismatch.size),
            },
        )
    return {"ok": True, "prefixes_checked": len(prefixes)}


def visible_count_matrix(
    records: Iterable[dict[str, Any]], prefixes: list[str], clocks: list[int]
) -> np.ndarray:
    """返回 shape=(len(prefixes), len(clocks)) 的可见计数矩阵。"""
    records = list(records)
    membership = np.zeros((len(prefixes), len(records)), dtype=np.int8)
    expires = np.array([r["expire_ms"] for r in records], dtype=np.int64)
    prefix_set = set(prefixes)
    for col, record in enumerate(records):
        for prefix in codec.prefixes_of(record["key"]):
            if prefix in prefix_set:
                membership[prefixes.index(prefix), col] = 1
    visible = np.zeros((len(prefixes), len(clocks)), dtype=np.int64)
    for j, clock in enumerate(clocks):
        # expire_ms == 0 永不过期；否则 clock < expire_ms 可见
        alive = (expires == 0) | (np.int64(clock) < expires)
        visible[:, j] = membership @ alive.astype(np.int8)
    return visible

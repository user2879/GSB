"""固定整数种子的本地确定性样本与 NumPy 定点分组聚合。

样本不依赖任何外部数据集：每个节点用声明中的整数种子通过
:class:`numpy.random.Generator` 生成固定形状的整数批次。聚合同样只用
整数运算（分组求和、计数、向零截断的整数均值），保证重放逐位一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GROUP_MOD = 8
VALUE_LOW = 0
VALUE_HIGH = 1_000_000


def generate_batch(seed: int, shard_id: str, sample_rows: int,
                   param_version: int) -> dict[str, Any]:
    """生成一批样本行：group 为小整数分组键，value 为非负整数。"""
    if sample_rows < 0:
        raise ValueError("sample_rows must be non-negative")
    # 混入分片标识与版本，使不同分片/版本的批次可区分且仍确定。
    shard_seed = _stable_mix(seed, shard_id, param_version)
    rng = np.random.default_rng(shard_seed)
    if sample_rows == 0:
        groups = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.int64)
    else:
        groups = rng.integers(0, GROUP_MOD, size=sample_rows, dtype=np.int64)
        values = rng.integers(VALUE_LOW, VALUE_HIGH, size=sample_rows,
                              dtype=np.int64)
    return {
        "shard_id": shard_id,
        "seed": int(seed),
        "param_version": int(param_version),
        "sample_rows": int(sample_rows),
        "groups": groups.tolist(),
        "values": values.tolist(),
    }


def _stable_mix(seed: int, shard_id: str, param_version: int) -> int:
    """把字符串分片标识稳定混入整数种子（FNV-1a 风格，纯整数）。"""
    digest = 0xCBF29CE484222325
    for char in shard_id.encode("utf-8"):
        digest ^= char
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    mixed = (seed * 0x9E3779B185EBCA87 + digest
             + param_version * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF
    return int(mixed)


@dataclass(frozen=True)
class AggregateSummary:
    shard_id: str
    node_id: str
    param_version: int
    row_count: int
    group_count: int
    group_sums: dict[str, int]
    group_counts: dict[str, int]
    group_means: dict[str, int]
    total_sum: int
    checksum: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "param_version": self.param_version,
            "row_count": self.row_count,
            "group_count": self.group_count,
            "group_sums": self.group_sums,
            "group_counts": self.group_counts,
            "group_means": self.group_means,
            "total_sum": self.total_sum,
            "checksum": self.checksum,
        }


def aggregate_batch(batch: dict[str, Any], node_id: str,
                    scale: int) -> AggregateSummary:
    """整数定点分组聚合；均值向零截断，缩放仅作用于均值量化。"""
    groups = np.asarray(batch["groups"], dtype=np.int64)
    values = np.asarray(batch["values"], dtype=np.int64)
    if groups.shape != values.shape:
        raise ValueError("groups and values must have the same length")
    group_sums: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    group_means: dict[str, int] = {}
    if groups.size:
        for group in np.unique(groups):
            mask = groups == group
            group_values = values[mask]
            count = int(group_values.size)
            total = int(group_values.sum(dtype=np.int64))
            key = str(int(group))
            group_sums[key] = total
            group_counts[key] = count
            # 定点均值：先量化再向零截断，全程整数。
            group_means[key] = (total * scale) // count
    total_sum = int(values.sum(dtype=np.int64)) if values.size else 0
    checksum = _checksum(group_sums, total_sum)
    return AggregateSummary(
        shard_id=batch["shard_id"], node_id=node_id,
        param_version=int(batch["param_version"]),
        row_count=int(groups.size), group_count=len(group_sums),
        group_sums=dict(sorted(group_sums.items())),
        group_counts=dict(sorted(group_counts.items())),
        group_means=dict(sorted(group_means.items())),
        total_sum=total_sum, checksum=checksum,
    )


def _checksum(group_sums: dict[str, int], total_sum: int) -> int:
    """聚合内容的整数摘要，用于账本与重放比对。"""
    acc = 0xCBF29CE484222325
    for key in sorted(group_sums):
        for char in key.encode("utf-8"):
            acc ^= char
            acc = (acc * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        acc ^= group_sums[key] & 0xFFFFFFFFFFFFFFFF
        acc = (acc * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    acc ^= total_sum & 0xFFFFFFFFFFFFFFFF
    return int((acc * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF)

"""Deterministic local sample generation."""

from __future__ import annotations

from typing import Any

import numpy as np

GROUP_COUNT = 4


def batch_seed(node_seed: int, shard_index: int, global_seed: int = 365) -> int:
    """Derive a stable non-negative 64-bit seed."""
    x = (int(global_seed) & 0xFFFF_FFFF) * 1_000_003
    x ^= (int(node_seed) & 0xFFFF_FFFF) * 91_763
    x ^= (int(shard_index) & 0xFFFF_FFFF) * 1_000_003
    x = (x + 0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF_FFFF
    x = ((x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    x = ((x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB) & 0xFFFF_FFFF_FFFF_FFFF
    return x ^ (x >> 31)


def generate_batch(seed: int, rows: int, group_count: int = GROUP_COUNT) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    row_count = int(rows)
    groups = rng.integers(0, group_count, size=row_count, dtype=np.int64)
    values = rng.integers(1, 1001, size=row_count, dtype=np.int64)
    weights = rng.integers(0, 10, size=row_count, dtype=np.int64)
    return {
        "seed": int(seed),
        "rows": row_count,
        "group_count": int(group_count),
        "groups": groups.tolist(),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }


def batch_digest(batch: dict[str, Any]) -> str:
    from .canonical import sha256_hex

    return sha256_hex(batch)


def aggregate_batch(batch: dict[str, Any], scale: int) -> dict[str, Any]:
    """Group and aggregate with NumPy; all returned sums are Python ints."""
    groups = np.asarray(batch["groups"], dtype=np.int64)
    values = np.asarray(batch["values"], dtype=np.int64)
    weights = np.asarray(batch.get("weights", np.ones_like(values)), dtype=np.int64)
    row_count = int(batch.get("rows", groups.shape[0]))
    group_count = int(batch.get("group_count", GROUP_COUNT))
    if groups.shape[0] != values.shape[0] or groups.shape[0] != weights.shape[0]:
        raise ValueError("sample array lengths differ")
    if row_count != groups.shape[0]:
        raise ValueError("row count does not match sample length")

    counts = np.bincount(groups, minlength=group_count)[:group_count].astype(np.int64)
    value_sums = np.zeros(group_count, dtype=np.int64)
    weighted_sums = np.zeros(group_count, dtype=np.int64)
    np.add.at(value_sums, groups, values)
    np.add.at(weighted_sums, groups, values * weights)
    total = int(value_sums.sum())
    weighted_total = int(weighted_sums.sum())
    scaled_total = total * int(scale)
    if scaled_total > 2**63 - 1:
        scaled_total = 2**63 - 1
    checksum_material = {
        "counts": counts.tolist(),
        "value_sums": value_sums.tolist(),
        "weighted_sums": weighted_sums.tolist(),
        "scale": scale,
    }
    from .canonical import sha256_hex

    return {
        "rows": row_count,
        "group_count": group_count,
        "counts": [int(value) for value in counts],
        "group_sums": [int(value) for value in value_sums],
        "weighted_group_sums": [int(value) for value in weighted_sums],
        "total": total,
        "weighted_total": weighted_total,
        "scale": int(scale),
        "scaled_total": int(scaled_total),
        "checksum": sha256_hex(checksum_material),
    }

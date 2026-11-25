from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import sha256_hex
from .timeutil import monotonic_ms


def block_group_checksum(profile: dict[str, Any]) -> dict[str, Any]:
    """Generate deterministic fixed-point samples and grouped NumPy sums."""
    started = monotonic_ms()
    seed = int(profile.get("seed", 20261126))
    sample_count = int(profile.get("sample_count", 64))
    group_count = int(profile.get("group_count", 4))
    scale = int(profile.get("scale", 1_000_000))

    rng = np.random.default_rng(seed)
    floating = rng.random(sample_count, dtype=np.float64)
    fixed_point = np.rint(floating * np.float64(scale)).astype(np.int64)
    groups = np.array_split(fixed_point, group_count)
    group_sums = [int(np.sum(group, dtype=np.int64)) for group in groups]
    total = sum(group_sums)
    digest = sha256_hex(
        {
            "algorithm": "numpy-fixed-point-group-sum-v1",
            "seed": seed,
            "sample_count": sample_count,
            "group_count": group_count,
            "scale": scale,
            "group_sums": group_sums,
            "total": total,
        }
    )
    ended = monotonic_ms()
    return {
        "algorithm": "numpy-fixed-point-group-sum-v1",
        "seed": seed,
        "sample_count": sample_count,
        "group_count": group_count,
        "scale": scale,
        "group_sums": group_sums,
        "total": total,
        "checksum_digest": digest,
        "compute_ms": max(0, int(ended - started)),
        "compute_precise_ms": round(ended - started, 3),
    }

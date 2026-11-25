from __future__ import annotations

from typing import Any

import numpy as np

from .spec import NodeSpec, ShardSpec, Spec


def sample_seed_for(spec: Spec, shard: ShardSpec, node: NodeSpec) -> int:
    """Derive a stable uint32 seed from the declaration's integer seeds."""

    return (
        int(spec.sample_seed) * 1_000_003
        + int(node.seed) * 9_176
        + shard.order_index * 7_919
    ) % (2**32)


def generate_samples(spec: Spec, shard: ShardSpec, node: NodeSpec) -> dict[str, Any]:
    seed = sample_seed_for(spec, shard, node)
    rng = np.random.default_rng(seed)
    group_count = max(1, min(8, max(1, shard.sample_rows // 12 or 1)))
    groups = rng.integers(0, group_count, size=shard.sample_rows, dtype=np.int64)
    values_q = rng.integers(-100_000, 100_001, size=shard.sample_rows, dtype=np.int64)
    return {
        "schema": "cloudshard.samples.v1",
        "sample_seed": seed,
        "declared_sample_seed": spec.sample_seed,
        "node_seed": node.seed,
        "row_count": shard.sample_rows,
        "groups": groups.tolist(),
        "values_q": values_q.tolist(),
    }

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from .graph import effective_node_id


def shard_seed(node_seed: int, shard_id: str) -> list[int]:
    return [int(node_seed) & 0xFFFFFFFF, zlib.crc32(shard_id.encode("utf-8"))]


def node_for_shard(spec: dict[str, Any], shard: dict[str, Any]) -> dict[str, Any]:
    node_id = effective_node_id(shard, spec)
    return spec["node_by_id"][node_id]


def generate_samples(spec: dict[str, Any], shard: dict[str, Any]) -> list[list[int]]:
    node = node_for_shard(spec, shard)
    group_count = int(shard.get("group_count") or node["group_count"])
    seed = shard_seed(node["seed"], shard["shard_id"])
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    rows = int(shard["rows"])
    groups = rng.integers(0, group_count, size=rows, dtype=np.int64)
    values = rng.integers(0, 10_000, size=rows, dtype=np.int64)
    return [[int(groups[i]), int(values[i])] for i in range(rows)]


def request_payload_for_shard(
    spec: dict[str, Any],
    shard: dict[str, Any],
    *,
    request_id: str,
    lease: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node = node_for_shard(spec, shard)
    return {
        "type": "aggregate",
        "request_id": request_id,
        "shard_id": shard["shard_id"],
        "node_id": node["node_id"],
        "capacity_version": int(node["capacity_version"]),
        "session_scale_factor": int(spec["scale_factor"]),
        "fixed_scale": int(node["fixed_scale"]),
        "lease": lease,
        "batch_samples": generate_samples(spec, shard),
    }

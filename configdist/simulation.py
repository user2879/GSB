from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import canonical_bytes, content_digest, sha256_bytes


MIN_SAMPLES = 128
MAX_SAMPLES = 256


def _sample_count(
    simulation_seed: int,
    cluster_id: str,
    revision_version: int,
    task_id: str,
    shard_id: str,
    config_keys: list[str],
    config_values: dict[str, Any],
) -> int:
    material = {
        "seed": simulation_seed,
        "cluster_id": cluster_id,
        "revision_version": revision_version,
        "task_id": task_id,
        "shard_id": shard_id,
        "config_keys": sorted(config_keys),
        "config_values": {key: config_values[key] for key in sorted(config_keys)},
    }
    offset = int(sha256_bytes(canonical_bytes(material))[:8], 16) % (MAX_SAMPLES - MIN_SAMPLES + 1)
    return MIN_SAMPLES + offset


def compute_aggregate(
    simulation_seed: int,
    cluster_id: str,
    revision_version: int,
    task_id: str,
    shard_id: str,
    config_keys: list[str],
    config_values: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic NumPy aggregate for one shard.

    The node process and the authoritative verifier call this exact function.
    A stable seed selects integer samples; referenced configuration keys define
    the groups and also affect the sample count.
    """

    keys = sorted(config_keys)
    if not keys:
        raise ValueError("config_keys must not be empty")
    missing = [key for key in keys if key not in config_values]
    if missing:
        raise ValueError(f"missing config values: {missing}")

    sample_count = _sample_count(
        simulation_seed,
        cluster_id,
        revision_version,
        task_id,
        shard_id,
        keys,
        config_values,
    )
    seed_material = canonical_bytes(
        {
            "simulation_seed": simulation_seed,
            "cluster_id": cluster_id,
            "revision_version": revision_version,
            "task_id": task_id,
            "shard_id": shard_id,
            "config_values": {key: config_values[key] for key in keys},
        }
    )
    rng_seed = int.from_bytes(sha256_bytes(seed_material).encode("ascii")[:16], "big")
    rng = np.random.default_rng(rng_seed)
    values = rng.integers(-1000, 1001, size=sample_count, dtype=np.int64)
    group_indexes = np.arange(sample_count, dtype=np.int64) % len(keys)

    groups: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        selected = values[group_indexes == index]
        groups.append(
            {
                "config_key": key,
                "count": int(selected.size),
                "sum": int(np.sum(selected, dtype=np.int64)),
                "min": int(np.min(selected)),
                "max": int(np.max(selected)),
            }
        )
    group_bytes = canonical_bytes([[group["config_key"], group["count"], group["sum"], group["min"], group["max"]] for group in groups])
    summary = {"scan_rows": int(sample_count), "groups": groups}
    return {
        "scan_rows": int(sample_count),
        "groups": groups,
        "byte_digest": sha256_bytes(group_bytes),
        "summary_digest": content_digest(summary),
    }

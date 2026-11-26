from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import fingerprint, stable_seed
from .models import BatchSpec, ShardSpec


def generate_batch(
    global_seed: int, shard: ShardSpec, batch: BatchSpec
) -> list[dict[str, Any]]:
    seed = stable_seed(global_seed, [shard.shard_id, batch.batch_id, str(shard.dimensions)])
    rng = np.random.Generator(np.random.PCG64(seed))
    group_indexes = rng.integers(0, len(batch.group_keys), size=batch.sample_count)
    values = rng.integers(
        -batch.value_bound,
        batch.value_bound + 1,
        size=(batch.sample_count, shard.dimensions),
        dtype=np.int64,
    )
    samples: list[dict[str, Any]] = []
    for index in range(batch.sample_count):
        samples.append(
            {
                "sample_id": f"{shard.shard_id}/{batch.batch_id}/{index}",
                "batch_id": batch.batch_id,
                "group": batch.group_keys[int(group_indexes[index])],
                "values": [int(value) for value in values[index].tolist()],
            }
        )
    return samples


def generate_shard_samples(global_seed: int, shard: ShardSpec) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for batch in sorted(shard.batches, key=lambda item: item.batch_id):
        samples.extend(generate_batch(global_seed, shard, batch))
    return samples


def samples_fingerprint(samples: list[dict[str, Any]]) -> str:
    canonical_samples = [
        {
            "sample_id": sample["sample_id"],
            "batch_id": sample["batch_id"],
            "group": sample["group"],
            "values": sample["values"],
        }
        for sample in sorted(samples, key=lambda item: str(item["sample_id"]))
    ]
    return fingerprint(canonical_samples)

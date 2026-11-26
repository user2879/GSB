from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .spec import SampleSpec, ShardSpec, canonical_json, sha256_text


def generate_integer_samples(seed: int, count: int = 8, value_count: int = 6) -> list[dict[str, Any]]:
    rng = np.random.RandomState(seed)
    samples: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        group_index = int(rng.randint(1, 4))
        values = [int(x) for x in rng.randint(0, 97, size=value_count, dtype=np.int64)]
        samples.append(
            {
                "id": f"sample-{seed}-{index:03d}",
                "group": f"g{group_index}",
                "seed": seed + index,
                "values": values,
            }
        )
    return samples


def sample_snapshot(samples: Iterable[SampleSpec | dict[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, SampleSpec):
            normalized.append({"id": sample.id, "group": sample.group, "values": list(sample.values)})
        else:
            normalized.append(
                {"id": sample["id"], "group": sample.get("group", "default"), "values": list(sample["values"])}
            )
    normalized.sort(key=lambda item: item["id"])
    return sha256_text(canonical_json(normalized))


def _fixed_mean(sum_value: int, count: int, scale: int) -> dict[str, int | bool]:
    if count <= 0:
        return {"q": 0, "remainder": 0, "denominator": 0, "saturated": True}
    raw = int(sum_value) * int(scale)
    q = raw // count
    remainder = raw - q * count
    saturated = q > 2**63 - 1
    if saturated:
        q = 2**63 - 1
    return {"q": int(q), "remainder": int(remainder), "denominator": int(count), "saturated": bool(saturated)}


def aggregate_shard(
    shard: ShardSpec | dict[str, Any],
    samples_by_id: dict[str, SampleSpec | dict[str, Any]],
    *,
    capacity_version: int,
    scale_factor: int,
    node_fixed_scale: int = 1,
) -> dict[str, Any]:
    """Deterministic NumPy grouped aggregation with an integer fixed-point mean."""

    if isinstance(shard, ShardSpec):
        shard_id = shard.id
        sample_ids = list(shard.sample_ids)
    else:
        shard_id = shard["id"]
        sample_ids = list(shard["sample_ids"])

    resolved: list[SampleSpec | dict[str, Any]] = [samples_by_id[sample_id] for sample_id in sample_ids]
    groups: dict[str, list[int]] = {}
    for sample in resolved:
        group = sample.group if isinstance(sample, SampleSpec) else str(sample.get("group", "default"))
        values = sample.values if isinstance(sample, SampleSpec) else sample["values"]
        groups.setdefault(group, []).extend(int(v) for v in values)

    group_results: dict[str, dict[str, Any]] = {}
    for group in sorted(groups):
        array = np.asarray(groups[group], dtype=np.int64)
        sum_value = int(np.sum(array, dtype=np.int64))
        count = int(array.size)
        minimum = int(np.min(array))
        maximum = int(np.max(array))
        mean = _fixed_mean(sum_value, count, node_fixed_scale)
        group_results[group] = {
            "count": count,
            "sum": sum_value,
            "min": minimum,
            "max": maximum,
            "fixed_mean_q": mean["q"],
            "fixed_mean_remainder": mean["remainder"],
            "fixed_mean_scale": node_fixed_scale,
            "saturated": mean["saturated"],
        }

    snapshot = sample_snapshot(resolved)
    canonical = {
        "shard_id": shard_id,
        "capacity_version": capacity_version,
        "session_scale_factor": scale_factor,
        "node_fixed_scale": node_fixed_scale,
        "groups": group_results,
        "sample_snapshot": snapshot,
    }
    summary = sha256_text(canonical_json(canonical))
    return {
        "shard_id": shard_id,
        "capacity_version": capacity_version,
        "scale_factor": scale_factor,
        "node_fixed_scale": node_fixed_scale,
        "groups": group_results,
        "sample_snapshot": snapshot,
        "summary": summary,
        "summary_prefix": summary[:16],
    }

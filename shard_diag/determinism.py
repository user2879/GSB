from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import AGGREGATION_CALIBER_VERSION
from .canonical import canonical_bytes, sha256_hex
from .spec import NodeSpec, ShardSpec

SAMPLE_FORMAT_VERSION = 1
GROUP_COUNT = 16
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def derive_sample_seed(node_seed: int, shard_id: str) -> int:
    material = f"shard-diag-sample-v1\n{int(node_seed)}\n{shard_id}".encode("utf-8")
    digest = sha256_hex(material)
    return int(digest[:8], 16)


def generate_samples(node_seed: int, shard_id: str, row_count: int) -> np.ndarray:
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    mixed_seed = derive_sample_seed(node_seed, shard_id)
    rng = np.random.RandomState(mixed_seed)
    groups = rng.randint(0, GROUP_COUNT, size=row_count).astype("<i8", copy=False)
    measures = rng.randint(-50_000, 50_000, size=row_count).astype("<i8", copy=False)
    return np.column_stack((groups, measures)).astype(np.int64, copy=False)


def samples_raw_bytes(samples: np.ndarray) -> bytes:
    """Stable little-endian bytes for both columns, independent of host byte order."""
    groups = np.ascontiguousarray(samples[:, 0], dtype="<i8")
    measures = np.ascontiguousarray(samples[:, 1], dtype="<i8")
    return groups.tobytes(order="C") + measures.tobytes(order="C")


def sample_artifact(node: NodeSpec, shard: ShardSpec) -> dict[str, Any]:
    samples = generate_samples(node.seed, shard.shard_id, shard.sample_rows)
    return {
        "format_version": SAMPLE_FORMAT_VERSION,
        "node_id": node.node_id,
        "shard_id": shard.shard_id,
        "node_seed": node.seed,
        "derived_seed": derive_sample_seed(node.seed, shard.shard_id),
        "row_count": shard.sample_rows,
        "group_count": GROUP_COUNT,
        "raw_columns": ["group_index:int64-le", "measure:int64-le"],
        "raw_bytes_sha256": sha256_hex(samples_raw_bytes(samples)),
        "rows": samples.astype(int).tolist(),
    }


def write_sample_artifact(path: str | Path, node: NodeSpec, shard: ShardSpec) -> dict[str, Any]:
    artifact = sample_artifact(node, shard)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(artifact) + b"\n")
    raw_path = path.with_suffix(".bin")
    samples = generate_samples(node.seed, shard.shard_id, shard.sample_rows)
    raw_path.write_bytes(samples_raw_bytes(samples))
    return artifact


def split_batches(samples: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(samples) == 0:
        return [samples]
    return [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]


def saturating_mul_int64(values: np.ndarray | list[int], multiplier: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    wide = arr.astype(object) * int(multiplier)
    result = np.empty(arr.shape, dtype=np.int64)
    high = wide > INT64_MAX
    low = wide < INT64_MIN
    safe = ~(high | low)
    result[high] = INT64_MAX
    result[low] = INT64_MIN
    if np.any(safe):
        result[safe] = wide[safe].astype(np.int64)
    return result


def empty_partial() -> dict[str, list[int]]:
    return {
        "counts": [0 for _ in range(GROUP_COUNT)],
        "sums_q": [0 for _ in range(GROUP_COUNT)],
    }


def aggregate_batch(batch: np.ndarray | list[list[int]], fixed_point_scale: int) -> dict[str, Any]:
    arr = np.asarray(batch, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("batch must have shape [n, 2]")
    if arr.shape[0] and (np.any(arr[:, 0] < 0) or np.any(arr[:, 0] >= GROUP_COUNT)):
        raise ValueError("group index is outside declared range")
    groups = arr[:, 0].astype(np.int64, copy=False)
    scaled_values = saturating_mul_int64(arr[:, 1], fixed_point_scale)
    counts = np.zeros(GROUP_COUNT, dtype=np.int64)
    sums = np.zeros(GROUP_COUNT, dtype=np.int64)
    np.add.at(counts, groups, 1)
    np.add.at(sums, groups, scaled_values)
    return {
        "counts": [int(value) for value in counts],
        "sums_q": [int(value) for value in sums],
        "participated_rows": int(counts.sum()),
    }


def combine_partials(partials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = [0] * GROUP_COUNT
    sums = [0] * GROUP_COUNT
    participated = 0
    for partial in partials:
        for index in range(GROUP_COUNT):
            counts[index] += int(partial["counts"][index])
            sums[index] += int(partial["sums_q"][index])
        participated += int(partial.get("participated_rows", sum(partial["counts"])))
    return {"counts": counts, "sums_q": sums, "participated_rows": participated}


def build_summary(
    *,
    node_id: str,
    shard_id: str,
    sample_rows: int,
    quota_version: int,
    partial: dict[str, Any],
    fixed_point_scale: int,
) -> dict[str, Any]:
    groups = [
        {"group": index, "count": int(partial["counts"][index]), "sum_q": int(partial["sums_q"][index])}
        for index in range(GROUP_COUNT)
    ]
    core = {
        "caliber_version": AGGREGATION_CALIBER_VERSION,
        "fixed_point_scale": int(fixed_point_scale),
        "node_id": node_id,
        "shard_id": shard_id,
        "sample_rows": int(sample_rows),
        "participated_rows": int(partial["participated_rows"]),
        "quota_version": int(quota_version),
        "group_count": GROUP_COUNT,
        "groups": groups,
    }
    return {**core, "aggregate_digest": sha256_hex(canonical_bytes(core))}


def expected_summary(
    node: NodeSpec,
    shard: ShardSpec,
    quota_version: int,
    *,
    batch_size: int | None = None,
    fixed_point_scale: int = 1_000_000,
) -> dict[str, Any]:
    samples = generate_samples(node.seed, shard.shard_id, shard.sample_rows)
    size = batch_size or shard.sample_rows or 1
    partial = combine_partials(
        aggregate_batch(batch, fixed_point_scale) for batch in split_batches(samples, size)
    )
    return build_summary(
        node_id=node.node_id,
        shard_id=shard.shard_id,
        sample_rows=shard.sample_rows,
        quota_version=quota_version,
        partial=partial,
        fixed_point_scale=fixed_point_scale,
    )


def summary_byte_key(summary: dict[str, Any]) -> bytes:
    """Canonical bytes excluding transport timing fields."""
    keys = [
        "caliber_version", "fixed_point_scale", "node_id", "shard_id",
        "sample_rows", "participated_rows", "quota_version", "group_count",
        "groups", "aggregate_digest",
    ]
    return canonical_bytes({key: summary[key] for key in keys})

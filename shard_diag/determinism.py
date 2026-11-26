from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from .canonical import digest_bytes
from .models import AggregateSummary, NodeSpec, ShardSpec


SAMPLE_DTYPE = np.dtype(
    [
        ("row_uid", "<u8"),
        ("group_a", "<i4"),
        ("group_b", "<i4"),
        ("metric", "<i8"),
        ("weight", "<i8"),
    ]
)


def derive_shard_seed(node_seed: int, shard_id: str) -> int:
    material = f"shard-diag/v1/{int(node_seed) & ((1 << 64) - 1)}/{shard_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    derived = int.from_bytes(digest[:8], "big", signed=False)
    return derived ^ (int(node_seed) & ((1 << 64) - 1))


def generate_samples(node_seed: int, shard_id: str, sample_rows: int) -> np.ndarray:
    if sample_rows < 0:
        raise ValueError("sample_rows must be non-negative")
    rows = int(sample_rows)
    if rows == 0:
        return np.zeros(0, dtype=SAMPLE_DTYPE)

    seed = derive_shard_seed(node_seed, shard_id)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(rows, dtype=SAMPLE_DTYPE)
    samples["row_uid"] = np.arange(rows, dtype=np.uint64)
    samples["group_a"] = rng.integers(0, 4, size=rows, dtype=np.int32)
    samples["group_b"] = rng.integers(0, 2, size=rows, dtype=np.int32)
    samples["metric"] = rng.integers(1, 101, size=rows, dtype=np.int64)
    samples["weight"] = rng.integers(1, 11, size=rows, dtype=np.int64)
    return samples


def sample_byte_digest(node_seed: int, shard_id: str, sample_rows: int) -> str:
    return digest_bytes(generate_samples(node_seed, shard_id, sample_rows).tobytes())


def aggregate_chunk(samples: np.ndarray, offset: int = 0, row_count: int | None = None) -> dict[str, Any]:
    total = len(samples)
    if offset < 0 or offset > total:
        raise ValueError("offset is outside sample array")
    end = total if row_count is None else offset + row_count
    if row_count is not None and (row_count < 0 or end > total):
        raise ValueError("offset/row_count extends outside sample array")

    subset = samples[offset:end]
    participating_rows = int(len(subset))
    if participating_rows == 0:
        return {"participating_rows": 0, "groups": []}

    group_a = subset["group_a"]
    group_b = subset["group_b"]
    order = np.lexsort((group_b, group_a))
    sorted_rows = subset[order]
    sorted_a = sorted_rows["group_a"]
    sorted_b = sorted_rows["group_b"]

    boundary = np.ones(participating_rows, dtype=bool)
    if participating_rows > 1:
        boundary[1:] = (sorted_a[1:] != sorted_a[:-1]) | (sorted_b[1:] != sorted_b[:-1])
    starts = np.flatnonzero(boundary)

    metric = sorted_rows["metric"]
    weight = sorted_rows["weight"]
    weighted_metric = metric * weight

    metric_sums = np.add.reduceat(metric, starts, dtype=np.int64)
    weight_sums = np.add.reduceat(weight, starts, dtype=np.int64)
    weighted_sums = np.add.reduceat(weighted_metric, starts, dtype=np.int64)
    counts = np.diff(starts, append=np.int64(participating_rows))

    groups: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        groups.append(
            {
                "group_a": int(sorted_a[start]),
                "group_b": int(sorted_b[start]),
                "row_count": int(counts[index]),
                "metric_sum": int(metric_sums[index]),
                "weight_sum": int(weight_sums[index]),
                "weighted_metric_sum": int(weighted_sums[index]),
            }
        )
    return {"participating_rows": participating_rows, "groups": groups}


def merge_chunks(chunks: Iterable[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    merged: dict[tuple[int, int], dict[str, int]] = {}
    participating_rows = 0
    for chunk in chunks:
        participating_rows += int(chunk["participating_rows"])
        for group in chunk["groups"]:
            key = (int(group["group_a"]), int(group["group_b"]))
            accumulator = merged.setdefault(
                key,
                {
                    "row_count": 0,
                    "metric_sum": 0,
                    "weight_sum": 0,
                    "weighted_metric_sum": 0,
                },
            )
            for field in ("row_count", "metric_sum", "weight_sum", "weighted_metric_sum"):
                accumulator[field] += int(group[field])

    groups: list[dict[str, Any]] = []
    for group_a, group_b in sorted(merged.keys()):
        values = merged[(group_a, group_b)]
        groups.append({"group_a": group_a, "group_b": group_b, **values})
    return participating_rows, groups


def partition_ranges(sample_rows: int, split_points: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """Return disjoint ``(offset, row_count)`` ranges covering all rows."""

    if split_points is None:
        if sample_rows <= 1:
            points: list[int] = []
        else:
            first = sample_rows // 3
            second = (2 * sample_rows) // 3
            points = sorted({point for point in (first, second) if 0 < point < sample_rows})
    else:
        points = sorted({int(point) for point in split_points})
        if any(point <= 0 or point >= sample_rows for point in points):
            raise ValueError("split points must be strictly inside [0, sample_rows]")
        if len(points) != len(set(split_points)):
            raise ValueError("split points must be unique")

    boundaries = [0, *points, sample_rows]
    ranges: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right > left:
            ranges.append((left, right - left))
    return ranges


def compute_shard_summary(
    node: NodeSpec,
    shard: ShardSpec,
    *,
    split_points: Sequence[int] | None = None,
) -> AggregateSummary:
    samples = generate_samples(node.seed, shard.shard_id, shard.sample_rows)
    digest = digest_bytes(samples.tobytes())
    ranges = partition_ranges(shard.sample_rows, split_points)
    chunks = [aggregate_chunk(samples, offset, count) for offset, count in ranges]
    participating_rows, groups = merge_chunks(chunks)
    return AggregateSummary(
        node_id=node.node_id,
        shard_id=shard.shard_id,
        sample_rows=shard.sample_rows,
        capacity_version=node.capacity_version,
        participating_rows=participating_rows,
        sample_digest=digest,
        groups=groups,
    )


def merge_partial_summaries(
    node: NodeSpec,
    shard: ShardSpec,
    sample_digest: str,
    partials: Sequence[dict[str, Any]],
) -> AggregateSummary:
    participating_rows, groups = merge_chunks(partials)
    return AggregateSummary(
        node_id=node.node_id,
        shard_id=shard.shard_id,
        sample_rows=shard.sample_rows,
        capacity_version=node.capacity_version,
        participating_rows=participating_rows,
        sample_digest=sample_digest,
        groups=groups,
    )

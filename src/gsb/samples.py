"""Deterministic integer samples and NumPy-only integer aggregation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

import numpy as np

from .model import NodeSpec, ShardSpec

SAMPLE_ALGORITHM = "gsb-pcg64-integer-v1"
AGGREGATE_SCHEMA = "gsb-fixed-group-sum-v1"
MAX_SAMPLE_VALUE = 10_000
VALUE_WIDTH = 3


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seed_words(node: NodeSpec, shard: ShardSpec) -> list[int]:
    shard_hash = hashlib.sha256(shard.id.encode("utf-8")).digest()
    return [
        node.seed,
        int.from_bytes(shard_hash[:8], "big"),
        int.from_bytes(shard_hash[8:16], "big"),
        shard.sample_count,
        shard.group_count,
        VALUE_WIDTH,
    ]


def generate_samples(node: NodeSpec, shard: ShardSpec) -> list[dict[str, Any]]:
    """Generate the declaration's rows exactly once, independent of batching."""

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(_seed_words(node, shard))))
    raw = rng.integers(
        -MAX_SAMPLE_VALUE,
        MAX_SAMPLE_VALUE + 1,
        size=(shard.sample_count, VALUE_WIDTH),
        dtype=np.int64,
    )
    group_ids = np.arange(shard.sample_count, dtype=np.int64) % shard.group_count
    return [
        {"g": int(group_ids[row_index]), "v": [int(value) for value in raw[row_index]]}
        for row_index in range(shard.sample_count)
    ]


def sample_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return canonical_json_bytes(list(rows))


def sample_digest(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_hex(sample_bytes(rows))


def _validated_arrays(rows: Sequence[dict[str, Any]], group_count: int) -> tuple[np.ndarray, np.ndarray]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if not rows:
        raise ValueError("at least one sample row is required")
    groups = np.empty(len(rows), dtype=np.int64)
    values = np.empty((len(rows), VALUE_WIDTH), dtype=np.int64)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"g", "v"}:
            raise ValueError(f"row {index} must contain exactly g and v")
        group = row["g"]
        vector = row["v"]
        if not isinstance(group, int) or isinstance(group, bool) or not 0 <= group < group_count:
            raise ValueError(f"row {index} has an invalid group")
        if (
            not isinstance(vector, list)
            or len(vector) != VALUE_WIDTH
            or any(not isinstance(value, int) or isinstance(value, bool) for value in vector)
        ):
            raise ValueError(f"row {index} has an invalid value vector")
        groups[index] = group
        values[index] = vector
    return groups, values


def aggregate_groups(rows: Sequence[dict[str, Any]], group_count: int, fixed_scale: int) -> dict[str, Any]:
    """Aggregate integer sample values using NumPy integer arrays only."""

    if fixed_scale <= 0:
        raise ValueError("fixed_scale must be positive")
    groups, values = _validated_arrays(rows, group_count)
    # Multiplication and accumulation stay inside int64.  Sample bounds make
    # this explicit check exact without floating-point conversion.
    max_abs = int(np.max(np.abs(values))) if values.size else 0
    if max_abs > ((1 << 63) - 1) // (fixed_scale * max(1, len(rows))):
        raise OverflowError("integer fixed-point sample aggregation would overflow int64")
    fixed_values = values * fixed_scale
    sums = np.zeros((group_count, VALUE_WIDTH), dtype=np.int64)
    counts = np.zeros(group_count, dtype=np.int64)
    np.add.at(sums, groups, fixed_values)
    np.add.at(counts, groups, 1)
    return {
        "groups": [
            {
                "group": group,
                "count": int(counts[group]),
                "sum_fixed": [int(value) for value in sums[group]],
            }
            for group in range(group_count)
        ],
        "participating_rows": len(rows),
    }


def merge_group_aggregates(group_count: int, aggregates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sums = np.zeros((group_count, VALUE_WIDTH), dtype=np.int64)
    counts = np.zeros(group_count, dtype=np.int64)
    participating_rows = 0
    seen_groups: set[int] = set()
    for aggregate in aggregates:
        if not isinstance(aggregate, dict) or "groups" not in aggregate:
            raise ValueError("invalid group aggregate")
        participating_rows += int(aggregate["participating_rows"])
        batch_groups: set[int] = set()
        for item in aggregate["groups"]:
            group = int(item["group"])
            count = int(item["count"])
            vector = item["sum_fixed"]
            if not 0 <= group < group_count or len(vector) != VALUE_WIDTH:
                raise ValueError("aggregate group is outside the declared range")
            if group in batch_groups:
                raise ValueError("aggregate repeats a group in one batch")
            batch_groups.add(group)
            seen_groups.add(group)
            sums[group] += np.asarray(vector, dtype=np.int64)
            counts[group] += count
    return {
        "groups": [
            {
                "group": group,
                "count": int(counts[group]),
                "sum_fixed": [int(value) for value in sums[group]],
            }
            for group in range(group_count)
        ],
        "participating_rows": participating_rows,
        "observed_groups": len(seen_groups),
    }


def final_aggregate(
    node: NodeSpec,
    shard: ShardSpec,
    capacity_version: str,
    rows: Sequence[dict[str, Any]],
    stage_aggregates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    merged = merge_group_aggregates(shard.group_count, stage_aggregates)
    if merged["participating_rows"] != shard.sample_count:
        raise ValueError("participating row count does not match the shard declaration")
    digest_payload = {
        "schema": AGGREGATE_SCHEMA,
        "sample_algorithm": SAMPLE_ALGORITHM,
        "node_id": node.id,
        "shard_id": shard.id,
        "sample_row_count": shard.sample_count,
        "capacity_version": capacity_version,
        "participating_rows": merged["participating_rows"],
        "fixed_scale": node.fixed_scale,
        "groups": merged["groups"],
        "sample_digest": sample_digest(rows),
    }
    merged["aggregate_digest"] = sha256_hex(canonical_json_bytes(digest_payload))
    merged["digest_payload"] = digest_payload
    merged["sample_digest"] = digest_payload["sample_digest"]
    merged["sample_byte_count"] = len(sample_bytes(rows))
    return merged


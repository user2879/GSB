from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np

from .hashing import sha256_hex


GROUP_COUNT = 4
SAMPLE_ALGORITHM = "seeded-shard-groups-v1"


def derive_sample_seed(node_seed: int, shard_id: str) -> int:
    """Derive one fixed 31-bit integer seed from node seed and shard id."""

    material = f"waterlease-v1:{int(node_seed)}:{shard_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def generate_samples(
    node_seed: int,
    shard_id: str,
    row_count: int,
    *,
    fixed_point_scale: int = 100,
) -> list[dict[str, int]]:
    """Generate deterministic integer sample rows.

    Only the declared node seed, shard id and row count influence the stream.
    The RNG is a fixed NumPy Generator seeded by a derived integer.
    """

    seed = derive_sample_seed(node_seed, shard_id)
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, GROUP_COUNT, size=row_count, dtype=np.int64)
    flags = rng.integers(0, 100, size=row_count, dtype=np.int64)
    values = rng.integers(0, 10_000, size=row_count, dtype=np.int64)
    scale_multiplier = max(1, fixed_point_scale // 100)
    rows: list[dict[str, int]] = []
    for index in range(row_count):
        rows.append(
            {
                "row_index": index,
                "group_id": int(groups[index]),
                "included": int(flags[index] >= 10),
                "value_q": int(values[index]) * scale_multiplier,
            }
        )
    return rows


def split_batches(rows: list[dict[str, int]], batch_size: int) -> list[list[dict[str, int]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def aggregate_samples(
    samples: Iterable[dict[str, Any]],
    *,
    fixed_point_scale: int = 100,
) -> dict[str, Any]:
    """Aggregate one batch with NumPy; all arithmetic stays integer."""

    rows = list(samples)
    sample_rows = len(rows)
    if sample_rows:
        group_ids = np.asarray([int(row["group_id"]) for row in rows], dtype=np.int64)
        included = np.asarray([int(row["included"]) != 0 for row in rows], dtype=bool)
        values = np.asarray([int(row["value_q"]) for row in rows], dtype=np.int64)
    else:
        group_ids = np.asarray([], dtype=np.int64)
        included = np.asarray([], dtype=bool)
        values = np.asarray([], dtype=np.int64)

    groups: list[dict[str, Any]] = []
    participated_rows = int(np.count_nonzero(included)) if sample_rows else 0
    for group_id in range(GROUP_COUNT):
        mask = included & (group_ids == group_id)
        count = int(np.count_nonzero(mask))
        if count:
            group_values = values[mask]
            total = int(group_values.sum(dtype=np.int64))
            minimum = int(group_values.min())
            maximum = int(group_values.max())
            mean_q = total // count
        else:
            total = 0
            minimum = None
            maximum = None
            mean_q = None
        groups.append(
            {
                "group_id": group_id,
                "participated_rows": count,
                "sum_value_q": total,
                "min_value_q": minimum,
                "max_value_q": maximum,
                "mean_value_q": mean_q,
                "rounding": "floor_nonnegative",
            }
        )

    return {
        "algorithm": SAMPLE_ALGORITHM,
        "fixed_point_scale": int(fixed_point_scale),
        "group_count": GROUP_COUNT,
        "sample_rows": sample_rows,
        "participated_rows": participated_rows,
        "groups": groups,
    }


def combine_aggregates(aggregates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combine batch monoids. Result is independent of batch order/splitting."""

    items = list(aggregates)
    if not items:
        raise ValueError("at least one aggregate is required")
    first = items[0]
    groups = [
        {
            "group_id": group_id,
            "participated_rows": 0,
            "sum_value_q": 0,
            "min_value_q": None,
            "max_value_q": None,
            "mean_value_q": None,
            "rounding": "floor_nonnegative",
        }
        for group_id in range(int(first["group_count"]))
    ]
    sample_rows = 0
    participated_rows = 0
    for aggregate in items:
        sample_rows += int(aggregate["sample_rows"])
        participated_rows += int(aggregate["participated_rows"])
        for batch_group in aggregate["groups"]:
            group_id = int(batch_group["group_id"])
            target = groups[group_id]
            count = int(batch_group["participated_rows"])
            target["participated_rows"] += count
            target["sum_value_q"] += int(batch_group["sum_value_q"])
            if count:
                low = batch_group["min_value_q"]
                high = batch_group["max_value_q"]
                target["min_value_q"] = low if target["min_value_q"] is None else min(target["min_value_q"], low)
                target["max_value_q"] = high if target["max_value_q"] is None else max(target["max_value_q"], high)

    for group in groups:
        count = group["participated_rows"]
        group["mean_value_q"] = group["sum_value_q"] // count if count else None

    return {
        "algorithm": SAMPLE_ALGORITHM,
        "fixed_point_scale": int(first["fixed_point_scale"]),
        "group_count": int(first["group_count"]),
        "sample_rows": sample_rows,
        "participated_rows": participated_rows,
        "groups": groups,
    }


def build_summary_core(
    *,
    node_id: str,
    shard_id: str,
    capacity_version: str,
    aggregate: dict[str, Any],
    upstream_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    upstream = dict(sorted((upstream_digests or {}).items()))
    core: dict[str, Any] = {
        "node_id": node_id,
        "shard_id": shard_id,
        "capacity_version": capacity_version,
        "sample_rows": int(aggregate["sample_rows"]),
        "participated_rows": int(aggregate["participated_rows"]),
        "fixed_point_scale": int(aggregate["fixed_point_scale"]),
        "groups": aggregate["groups"],
        "upstream_digests": [
            {"shard_id": shard_id, "aggregate_digest": digest}
            for shard_id, digest in upstream.items()
        ],
    }
    core["aggregate_digest"] = sha256_hex(
        {key: value for key, value in core.items() if key != "aggregate_digest"}
    )
    return core

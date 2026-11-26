from __future__ import annotations

import hashlib
import struct
from typing import Any, Iterable, Sequence

import numpy as np

from .canonical import content_digest

GROUP_COUNT = 8
SUMMARY_SCHEMA_VERSION = 1
MAX_FIXED_POINT = 2**63 - 1


def shard_entropy(node_seed: int, shard_id: str) -> list[int]:
    material = f"{int(node_seed)}\x1f{shard_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return list(struct.unpack(">8I", digest))


def generate_sample_arrays(
    node_seed: int, shard_id: str, sample_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    sequence = np.random.SeedSequence(shard_entropy(node_seed, shard_id))
    rng = np.random.default_rng(sequence)
    groups = rng.integers(0, GROUP_COUNT, size=int(sample_rows), dtype=np.int64)
    values = rng.integers(0, 1000, size=int(sample_rows), dtype=np.int64)
    return groups, values


def arrays_to_wire(groups: np.ndarray, values: np.ndarray) -> list[list[int]]:
    return [[int(g), int(v)] for g, v in zip(groups, values)]


def generate_samples(node_seed: int, shard_id: str, sample_rows: int) -> list[list[int]]:
    groups, values = generate_sample_arrays(node_seed, shard_id, sample_rows)
    return arrays_to_wire(groups, values)


def sample_digest(node_seed: int, shard_id: str, sample_rows: int) -> str:
    groups, values = generate_sample_arrays(node_seed, shard_id, sample_rows)
    return hashlib.sha256(
        groups.astype(">i8", copy=False).tobytes(order="C")
        + values.astype(">i8", copy=False).tobytes(order="C")
    ).hexdigest()


def _saturate(value: int) -> int:
    if value > MAX_FIXED_POINT:
        return MAX_FIXED_POINT
    if value < -MAX_FIXED_POINT - 1:
        return -MAX_FIXED_POINT - 1
    return int(value)


def aggregate_samples_numpy(
    samples: Sequence[Sequence[int]],
    *,
    fixed_point_exponent: int = 0,
) -> list[dict[str, int]]:
    """Exact group aggregates. NumPy performs unique/count/at reductions."""

    if not samples:
        return []
    arr = np.asarray(samples, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("samples must be rows of [group, value]")
    groups = arr[:, 0].astype(np.int64)
    values = arr[:, 1].astype(np.int64)
    if np.any((groups < 0) | (groups >= GROUP_COUNT)):
        raise ValueError("sample group out of range")

    counts = np.bincount(groups, minlength=GROUP_COUNT).astype(np.int64)
    sums = np.zeros(GROUP_COUNT, dtype=object)
    sum_squares = np.zeros(GROUP_COUNT, dtype=object)
    minima = np.full(GROUP_COUNT, np.iinfo(np.int64).max, dtype=np.int64)
    maxima = np.full(GROUP_COUNT, np.iinfo(np.int64).min, dtype=np.int64)

    values_obj = values.astype(object)
    np.add.at(sums, groups, values_obj)
    np.add.at(sum_squares, groups, values_obj * values_obj)
    np.minimum.at(minima, groups, values)
    np.maximum.at(maxima, groups, values)

    factor = 10 ** int(fixed_point_exponent)
    rows: list[dict[str, int]] = []
    for group in np.unique(groups):
        idx = int(group)
        row_count = int(counts[idx])
        total = int(sums[idx])
        mean_fp = _saturate((total * factor) // row_count)
        rows.append(
            {
                "group": idx,
                "rows": row_count,
                "sum": total,
                "sum_sq": int(sum_squares[idx]),
                "min": int(minima[idx]),
                "max": int(maxima[idx]),
                "mean_fp": mean_fp,
                "fixed_point_factor": factor,
            }
        )
    return rows


def make_batch_summary(
    *,
    node_id: str,
    shard_id: str,
    capacity_version: int,
    samples: Sequence[Sequence[int]],
    fixed_point_exponent: int = 0,
    batch_index: int = 0,
) -> dict[str, Any]:
    groups = aggregate_samples_numpy(
        samples, fixed_point_exponent=fixed_point_exponent
    )
    participating = sum(row["rows"] for row in groups)
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "batch_summary",
        "node_id": node_id,
        "shard_id": shard_id,
        "capacity_version": int(capacity_version),
        "fixed_point_exponent": int(fixed_point_exponent),
        "batch_index": int(batch_index),
        "batch_rows": len(samples),
        "participating_rows": participating,
        "groups": groups,
    }
    summary["batch_digest"] = content_digest(summary)
    return summary


def merge_group_rows(
    left: list[dict[str, int]],
    right: list[dict[str, int]],
    fp_exp: int,
) -> list[dict[str, int]]:
    by_group = {int(row["group"]): dict(row) for row in left}
    for raw_row in right:
        row = dict(raw_row)
        idx = int(row["group"])
        if idx not in by_group:
            by_group[idx] = row
            continue
        target = by_group[idx]
        target["rows"] = int(target["rows"]) + int(row["rows"])
        target["sum"] = int(target["sum"]) + int(row["sum"])
        target["sum_sq"] = int(target["sum_sq"]) + int(row["sum_sq"])
        target["min"] = min(int(target["min"]), int(row["min"]))
        target["max"] = max(int(target["max"]), int(row["max"]))

    factor = 10 ** int(fp_exp)
    merged: list[dict[str, int]] = []
    for idx in sorted(by_group):
        row = by_group[idx]
        count = int(row["rows"])
        total = int(row["sum"])
        merged.append(
            {
                "group": idx,
                "rows": count,
                "sum": total,
                "sum_sq": int(row["sum_sq"]),
                "min": int(row["min"]),
                "max": int(row["max"]),
                "mean_fp": _saturate((total * factor) // count),
                "fixed_point_factor": factor,
            }
        )
    return merged


def merge_batch_summaries(
    batches: Iterable[dict[str, Any]],
    *,
    node_id: str,
    shard_id: str,
    sample_rows: int,
    capacity_version: int,
    fixed_point_exponent: int,
    full_sample_digest: str,
) -> dict[str, Any]:
    ordered = sorted(batches, key=lambda item: int(item.get("batch_index", 0)))
    groups: list[dict[str, int]] = []
    participating = 0
    for batch in ordered:
        if batch.get("node_id") != node_id or batch.get("shard_id") != shard_id:
            raise ValueError("batch identity mismatch")
        if int(batch.get("capacity_version")) != int(capacity_version):
            raise ValueError("batch capacity version mismatch")
        groups = merge_group_rows(
            groups, batch.get("groups", []), fixed_point_exponent
        )
        participating += int(batch.get("participating_rows", 0))

    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "shard_summary",
        "node_id": node_id,
        "shard_id": shard_id,
        "sample_rows": int(sample_rows),
        "participating_rows": int(participating),
        "capacity_version": int(capacity_version),
        "fixed_point_exponent": int(fixed_point_exponent),
        "sample_digest": full_sample_digest,
        "groups": groups,
    }
    if participating != sample_rows:
        raise ValueError(
            f"participating_rows {participating} != sample_rows {sample_rows}"
        )
    summary["summary_digest"] = content_digest(
        {key: value for key, value in summary.items() if key != "summary_digest"}
    )
    return summary

"""Deterministic sample generation and integer fixed-point aggregation."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from .canonical import canonical_digest


SAMPLE_VERSION = "sharddiag-samples-v1"
GROUP_COUNT = 8


def shard_seed_words(node_seed: int, shard_id: str) -> list[int]:
    """Derive stable 32-bit seed words from node seed and shard id."""

    material = f"{SAMPLE_VERSION}\n{int(node_seed)}\n{shard_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 32, 4)]


def make_rng(node_seed: int, shard_id: str) -> np.random.Generator:
    words = shard_seed_words(node_seed, shard_id)
    seed = np.random.SeedSequence([int(node_seed) & 0xFFFFFFFF, *words])
    return np.random.Generator(np.random.PCG64(seed))


def generate_samples(node_seed: int, shard_id: str, rows: int) -> dict[str, np.ndarray]:
    """Generate bit-stable integer sample columns.

    ``value_q`` is already expressed at the node's decimal fixed-point scale.
    All generated weights are positive, so every row participates.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    rng = make_rng(node_seed, shard_id)
    return {
        "group_id": rng.integers(0, GROUP_COUNT, size=rows, dtype=np.int64),
        "value_q": rng.integers(-10_000, 10_001, size=rows, dtype=np.int64),
        "weight": rng.integers(1, 10, size=rows, dtype=np.int64),
    }


def sample_digest(samples: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(SAMPLE_VERSION.encode("ascii"))
    for name in ("group_id", "value_q", "weight"):
        array = np.ascontiguousarray(samples[name], dtype="<i8")
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape[0]).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def samples_to_wire(samples: dict[str, np.ndarray], start: int = 0, end: int | None = None) -> dict[str, list[int]]:
    if end is None:
        end = int(samples["group_id"].shape[0])
    return {
        "group_id": [int(value) for value in samples["group_id"][start:end]],
        "value_q": [int(value) for value in samples["value_q"][start:end]],
        "weight": [int(value) for value in samples["weight"][start:end]],
    }


def samples_from_wire(payload: dict[str, Sequence[int]]) -> dict[str, np.ndarray]:
    return {
        "group_id": np.asarray(payload["group_id"], dtype=np.int64),
        "value_q": np.asarray(payload["value_q"], dtype=np.int64),
        "weight": np.asarray(payload["weight"], dtype=np.int64),
    }


def _floor_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return divmod(int(numerator), int(denominator))[0]


def aggregate_partial(
    samples: dict[str, np.ndarray],
    *,
    fixed_point_decimals: int,
    batch_index: int = 0,
) -> dict[str, Any]:
    """Aggregate one contiguous batch with NumPy integer operations."""

    groups = np.asarray(samples["group_id"], dtype=np.int64)
    values = np.asarray(samples["value_q"], dtype=np.int64)
    weights = np.asarray(samples["weight"], dtype=np.int64)
    if not (groups.shape == values.shape == weights.shape):
        raise ValueError("sample columns must have equal shapes")

    row_count = int(groups.shape[0])
    participating = int(np.count_nonzero(weights > 0))
    counts = np.bincount(groups, minlength=GROUP_COUNT).astype(np.int64, copy=False)
    sums = np.zeros(GROUP_COUNT, dtype=np.int64)
    weighted_sums = np.zeros(GROUP_COUNT, dtype=np.int64)
    weight_sums = np.zeros(GROUP_COUNT, dtype=np.int64)
    minima = np.full(GROUP_COUNT, np.iinfo(np.int64).max, dtype=np.int64)
    maxima = np.full(GROUP_COUNT, np.iinfo(np.int64).min, dtype=np.int64)

    products = values * weights
    np.add.at(sums, groups, values)
    np.add.at(weighted_sums, groups, products)
    np.add.at(weight_sums, groups, weights)
    for group_id in range(GROUP_COUNT):
        mask = groups == group_id
        if np.any(mask):
            minima[group_id] = int(np.min(values[mask]))
            maxima[group_id] = int(np.max(values[mask]))

    group_rows: list[dict[str, int | None]] = []
    for group_id in range(GROUP_COUNT):
        count = int(counts[group_id])
        weight_sum = int(weight_sums[group_id])
        sum_q = int(sums[group_id])
        weighted_sum_q = int(weighted_sums[group_id])
        if count:
            mean_q_floor = _floor_div(sum_q, count)
            weighted_mean_q_floor = _floor_div(weighted_sum_q, weight_sum)
            minimum_q: int | None = int(minima[group_id])
            maximum_q: int | None = int(maxima[group_id])
        else:
            mean_q_floor = 0
            weighted_mean_q_floor = 0
            minimum_q = None
            maximum_q = None
        group_rows.append(
            {
                "group_id": group_id,
                "count": count,
                "sum_q": sum_q,
                "mean_q_floor": mean_q_floor,
                "weight_sum": weight_sum,
                "weighted_sum_q": weighted_sum_q,
                "weighted_mean_q_floor": weighted_mean_q_floor,
                "min_q": minimum_q,
                "max_q": maximum_q,
            }
        )

    return {
        "aggregation_version": SAMPLE_VERSION,
        "batch_index": int(batch_index),
        "fixed_point_decimals": int(fixed_point_decimals),
        "row_count": row_count,
        "participating_rows": participating,
        "groups": group_rows,
    }


def merge_partials(partials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge commutative partial group aggregates and recompute means."""

    merged_groups: dict[int, dict[str, Any]] = {
        group_id: {
            "group_id": group_id,
            "count": 0,
            "sum_q": 0,
            "mean_q_floor": 0,
            "weight_sum": 0,
            "weighted_sum_q": 0,
            "weighted_mean_q_floor": 0,
            "min_q": None,
            "max_q": None,
        }
        for group_id in range(GROUP_COUNT)
    }
    total_rows = 0
    participating = 0
    decimals: int | None = None
    version: str | None = None
    for partial in partials:
        if decimals is None:
            decimals = int(partial["fixed_point_decimals"])
        elif decimals != int(partial["fixed_point_decimals"]):
            raise ValueError("cannot merge different fixed-point calibers")
        version = str(partial.get("aggregation_version", SAMPLE_VERSION))
        total_rows += int(partial["row_count"])
        participating += int(partial["participating_rows"])
        for row in partial["groups"]:
            group_id = int(row["group_id"])
            target = merged_groups[group_id]
            target["count"] += int(row["count"])
            target["sum_q"] += int(row["sum_q"])
            target["weight_sum"] += int(row["weight_sum"])
            target["weighted_sum_q"] += int(row["weighted_sum_q"])
            if row["min_q"] is not None:
                target["min_q"] = row["min_q"] if target["min_q"] is None else min(target["min_q"], row["min_q"])
            if row["max_q"] is not None:
                target["max_q"] = row["max_q"] if target["max_q"] is None else max(target["max_q"], row["max_q"])

    for target in merged_groups.values():
        if target["count"]:
            target["mean_q_floor"] = _floor_div(target["sum_q"], target["count"])
            target["weighted_mean_q_floor"] = _floor_div(
                target["weighted_sum_q"], target["weight_sum"]
            )

    return {
        "aggregation_version": version or SAMPLE_VERSION,
        "fixed_point_decimals": decimals or 0,
        "row_count": total_rows,
        "participating_rows": participating,
        "groups": [merged_groups[group_id] for group_id in range(GROUP_COUNT)],
    }


def bind_summary(
    merged: dict[str, Any],
    *,
    node_id: str,
    shard_id: str,
    sample_rows: int,
    capacity_version: int,
    digest_of_samples: str,
    batch_size: int,
) -> dict[str, Any]:
    """Add identity bindings and the canonical summary digest."""

    summary = {
        **merged,
        "node_id": node_id,
        "shard_id": shard_id,
        "sample_rows": int(sample_rows),
        "capacity_version": int(capacity_version),
        "sample_digest": digest_of_samples,
        "group_count": GROUP_COUNT,
        "mean_rounding": "floor_divmod_toward_negative_infinity",
    }
    summary["summary_digest"] = canonical_digest({key: value for key, value in summary.items() if key != "summary_digest"})
    return summary


def derivation_basis(
    *,
    node_id: str,
    node_seed: int,
    shard_id: str,
    sample_rows: int,
    capacity_version: int,
    fixed_point_decimals: int,
    batch_size: int,
    samples: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> dict[str, Any]:
    first_rows = min(16, sample_rows)
    return {
        "rule": (
            "sha256('sharddiag-samples-v1\\n<node_seed>\\n<shard_id>') supplies eight "
            "big-endian uint32 SeedSequence words; NumPy PCG64/Generator draws group_id "
            "[0,8), value_q [-10000,10001), and weight [1,10). Aggregation uses bincount/"
            "add.at integer sums and Python divmod floor means; partials are commutative."
        ),
        "node_id": node_id,
        "shard_id": shard_id,
        "node_seed": int(node_seed),
        "seed_words": shard_seed_words(node_seed, shard_id),
        "sample_rows": int(sample_rows),
        "capacity_version": int(capacity_version),
        "fixed_point_decimals": int(fixed_point_decimals),
        "fixed_point_scale": 10 ** int(fixed_point_decimals),
        "batch_size": int(batch_size),
        "sample_digest": sample_digest(samples),
        "first_rows": samples_to_wire(samples, 0, first_rows),
        "numpy_version": np.__version__,
        "summary_digest": summary["summary_digest"],
        "participating_rows": summary["participating_rows"],
    }


def compute_shard_summary_local(
    *,
    node_id: str,
    node_seed: int,
    shard_id: str,
    sample_rows: int,
    capacity_version: int,
    fixed_point_decimals: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Convenience deterministic computation used by tests and local tooling."""

    samples = generate_samples(node_seed, shard_id, sample_rows)
    partials = []
    for batch_index, start in enumerate(range(0, sample_rows, batch_size)):
        end = min(sample_rows, start + batch_size)
        part = {name: array[start:end] for name, array in samples.items()}
        partials.append(
            aggregate_partial(
                part,
                fixed_point_decimals=fixed_point_decimals,
                batch_index=batch_index,
            )
        )
    merged = merge_partials(partials)
    summary = bind_summary(
        merged,
        node_id=node_id,
        shard_id=shard_id,
        sample_rows=sample_rows,
        capacity_version=capacity_version,
        digest_of_samples=sample_digest(samples),
        batch_size=batch_size,
    )
    basis = derivation_basis(
        node_id=node_id,
        node_seed=node_seed,
        shard_id=shard_id,
        sample_rows=sample_rows,
        capacity_version=capacity_version,
        fixed_point_decimals=fixed_point_decimals,
        batch_size=batch_size,
        samples=samples,
        summary=summary,
    )
    return samples, summary, basis

"""Deterministic integer sample generation and fixed-point group aggregation.

The rules below are intentionally explicit and order independent so that the
same declaration produces bit-identical sample bytes and aggregate summaries
under any dispatch ordering, batch split or concurrency level.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
GROUP_CARDINALITY = 16


def stable_hash64(token: str) -> int:
    """FNV-1a 64-bit hash over the UTF-8 bytes of ``token``."""
    value = FNV_OFFSET
    for byte in token.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & MASK64
    return value


def rng_seed_pair(seed: int, shard_id: str) -> tuple[int, int]:
    """Derive two explicit uint64 entropy words from node seed and shard id."""
    first = stable_hash64(f"{seed}|{shard_id}|group")
    second = stable_hash64(f"{seed}|{shard_id}|value")
    return first, second


def make_rng(seed: int, shard_id: str) -> np.random.Generator:
    first, second = rng_seed_pair(seed, shard_id)
    seed_sequence = np.random.SeedSequence([first, second])
    return np.random.Generator(np.random.SFC64(seed_sequence))


def generate_sample(seed: int, shard_id: str, rows: int) -> np.ndarray:
    """Generate ``rows`` deterministic (group, value) int64 pairs.

    Groups span 0..15 and values span -1000..1000 inclusive. A row participates
    in aggregation only when its value is non-negative; the participation rule
    is an explicit fixed-point filter, not a runtime decision.
    """
    rng = make_rng(seed, shard_id)
    groups = rng.integers(0, GROUP_CARDINALITY, size=rows, dtype=np.int64)
    values = rng.integers(-1000, 1001, size=rows, dtype=np.int64)
    return np.column_stack((groups, values)).astype(np.int64, copy=False)


def sample_bytes(seed: int, shard_id: str, rows: int) -> bytes:
    """Canonical little-endian int64 sample bytes (row-major group, value)."""
    return generate_sample(seed, shard_id, rows).tobytes()


def aggregate_rows(data: np.ndarray) -> dict:
    """Aggregate a 2-D int64 array of (group, value) rows with NumPy.

    Participating rows are those with value >= 0. Per group we report count,
    signed int64 sum, minimum and maximum, each derived by vectorized
    reductions. Empty groups are omitted so merging batches stays order free.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("aggregate_rows expects an (N, 2) int64 array")
    array = np.asarray(data, dtype=np.int64)
    participating = array[array[:, 1] >= 0]
    if participating.shape[0] == 0:
        return {"participating_rows": 0, "groups": {}}

    order = np.argsort(participating[:, 0], kind="stable")
    ordered = participating[order]
    sorted_groups = ordered[:, 0]
    sorted_values = ordered[:, 1]
    boundary_index = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    group_ids = sorted_groups[boundary_index]
    counts = np.diff(np.r_[boundary_index, sorted_groups.shape[0]])
    sums = np.add.reduceat(sorted_values, boundary_index)
    minimums = np.minimum.reduceat(sorted_values, boundary_index)
    maximums = np.maximum.reduceat(sorted_values, boundary_index)

    groups: dict[str, dict[str, int]] = {}
    for group_id, count, total, minimum, maximum in zip(
        group_ids, counts, sums, minimums, maximums
    ):
        groups[str(int(group_id))] = {
            "count": int(count),
            "sum": int(total),
            "min": int(minimum),
            "max": int(maximum),
        }
    return {"participating_rows": int(participating.shape[0]), "groups": groups}


def merge_aggregate(left: dict, right: dict) -> dict:
    """Merge two aggregate summaries; associative and commutative for int64."""
    groups = {key: dict(value) for key, value in left.get("groups", {}).items()}
    for key, value in right.get("groups", {}).items():
        if key not in groups:
            groups[key] = dict(value)
            continue
        current = groups[key]
        current["count"] += value["count"]
        current["sum"] += value["sum"]
        current["min"] = min(current["min"], value["min"])
        current["max"] = max(current["max"], value["max"])
    merged = {
        "participating_rows": int(left.get("participating_rows", 0))
        + int(right.get("participating_rows", 0)),
        "groups": dict(sorted(groups.items(), key=lambda item: int(item[0]))),
    }
    return merged


def canonical_json(payload: object) -> bytes:
    """Deterministic JSON byte encoding used for digests and byte equality."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()

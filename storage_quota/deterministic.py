from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import canonical_bytes, sha256_hex

GROUP_COUNT = 16
POINT_SCALE = 1000
DIGEST_SCHEMA = "sq.aggregate.v1"


def derive_pcg_seed(node_seed: int, shard_id: str) -> int:
    material = f"sq.sample.v1\0{node_seed}\0{shard_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def generate_rows(node_seed: int, shard_id: str, start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (group, value_thousandths) for a deterministic row interval.

    Row i is the i-th uint64 drawn by PCG64 seeded from SHA-256(node_seed, shard_id).
    The low four bits select one of 16 groups and the next ten bits select an
    integer value in thousandths.  PCG64.advance makes a slice independent of
    batch boundaries and batch arrival order.
    """
    if start < 0 or count < 0:
        raise ValueError("row interval must be non-negative")
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    bitgen = np.random.PCG64(derive_pcg_seed(node_seed, shard_id))
    bitgen.advance(start)
    rng = np.random.Generator(bitgen)
    raw = rng.integers(0, 2**63, size=count, dtype=np.uint64)
    groups = (raw % np.uint64(GROUP_COUNT)).astype(np.int64)
    values = ((raw // np.uint64(GROUP_COUNT)) % np.uint64(POINT_SCALE)).astype(np.int64)
    return groups, values


@dataclass
class AggregateAccumulator:
    node_id: str
    shard_id: str
    node_seed: int
    rows: int
    capacity_version: int
    counts: np.ndarray
    sums: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    participated_rows: int = 0

    @classmethod
    def create(cls, *, node_id: str, shard_id: str, node_seed: int, rows: int,
               capacity_version: int) -> "AggregateAccumulator":
        if rows <= 0:
            raise ValueError("rows must be positive")
        return cls(
            node_id=node_id,
            shard_id=shard_id,
            node_seed=node_seed,
            rows=rows,
            capacity_version=capacity_version,
            counts=np.zeros(GROUP_COUNT, dtype=np.int64),
            sums=np.zeros(GROUP_COUNT, dtype=np.int64),
            mins=np.full(GROUP_COUNT, np.iinfo(np.int64).max, dtype=np.int64),
            maxs=np.full(GROUP_COUNT, np.iinfo(np.int64).min, dtype=np.int64),
        )

    def add_interval(self, start: int, count: int) -> int:
        if start < 0 or count < 0 or start + count > self.rows:
            raise ValueError("batch interval outside shard")
        groups, values = generate_rows(self.node_seed, self.shard_id, start, count)
        if count:
            nonzero = np.bincount(groups, minlength=GROUP_COUNT).astype(np.int64)
            self.counts += nonzero
            np.add.at(self.sums, groups, values)
            for group in np.flatnonzero(nonzero):
                group_values = values[groups == group]
                self.mins[group] = min(int(self.mins[group]), int(group_values.min()))
                self.maxs[group] = max(int(self.maxs[group]), int(group_values.max()))
            self.participated_rows += count
        return self.participated_rows

    def summary(self) -> dict[str, Any]:
        groups = []
        for group in range(GROUP_COUNT):
            count = int(self.counts[group])
            if count:
                minimum = int(self.mins[group])
                maximum = int(self.maxs[group])
            else:
                minimum = 0
                maximum = 0
            groups.append({
                "group": group,
                "count": count,
                "sum_q": int(self.sums[group]),
                "min_q": minimum,
                "max_q": maximum,
            })
        return {
            "schema": DIGEST_SCHEMA,
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "rows": self.rows,
            "capacity_version": self.capacity_version,
            "point_scale": POINT_SCALE,
            "participated_rows": self.participated_rows,
            "groups": groups,
        }

    def digest(self) -> str:
        return sha256_hex(self.summary())

    def summary_bytes(self) -> bytes:
        return canonical_bytes(self.summary())


def aggregate_full(*, node_id: str, shard_id: str, node_seed: int, rows: int,
                   capacity_version: int, batch_size: int = 131072) -> tuple[dict[str, Any], str]:
    accumulator = AggregateAccumulator.create(
        node_id=node_id,
        shard_id=shard_id,
        node_seed=node_seed,
        rows=rows,
        capacity_version=capacity_version,
    )
    for start in range(0, rows, batch_size):
        accumulator.add_interval(start, min(batch_size, rows - start))
    return accumulator.summary(), accumulator.digest()

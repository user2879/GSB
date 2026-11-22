"""Deterministic sample generation and NumPy group aggregation.

Seeds are derived from a stable FNV-1a hash of (base_seed, node_id, item_key)
so that:

* the same logical request always yields identical samples on every machine
  and every run;
* different (node, item) pairs draw independent streams without dict/order
  dependence.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_AGGREGATIONS = ("sum", "mean", "min", "max")
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def stable_hash(*parts: object) -> int:
    """Platform-independent 64-bit FNV-1a hash over stringified parts."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = _FNV_OFFSET
    for byte in payload:
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _MASK64
    return digest


def sample_stream(base_seed: int, node_id: str, item_key: str):
    seed_hash = stable_hash(base_seed, node_id, item_key)
    ss = np.random.SeedSequence([seed_hash & 0xFFFFFFFF,
                                 (seed_hash >> 32) & 0xFFFFFFFF])
    return np.random.default_rng(ss)


def generate_samples(base_seed: int, node_id: str, item_key: str,
                     sample_count: int, group_count: int):
    """Return ``(group_keys, raw_values)`` for one (node, item) pair."""

    rng = sample_stream(base_seed, node_id, item_key)
    group_keys = rng.integers(0, group_count, size=sample_count,
                              dtype=np.int64)
    raw = rng.normal(loc=0.0, scale=1.0, size=sample_count)
    return group_keys, raw


def aggregate(group_keys: np.ndarray, values: np.ndarray,
              aggregations: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Group-wise aggregation; group keys emitted as sorted decimal strings."""

    summary: dict[str, dict[str, float]] = {}
    unique_groups = np.unique(group_keys)
    for group in unique_groups:
        bucket = values[group_keys == group]
        record: dict[str, float] = {}
        for name in aggregations:
            if name == "sum":
                record[name] = float(np.sum(bucket))
            elif name == "mean":
                record[name] = float(np.mean(bucket))
            elif name == "min":
                record[name] = float(np.min(bucket))
            elif name == "max":
                record[name] = float(np.max(bucket))
            else:  # pragma: no cover - validated at session level
                raise ValueError(f"unsupported aggregation {name}")
        summary[str(int(group))] = record
    return dict(sorted(summary.items()))


def run_aggregation(base_seed: int, node_id: str, item_key: str,
                    sample_count: int, group_count: int,
                    aggregations: tuple[str, ...],
                    scale_value: float) -> dict[str, dict[str, float]]:
    group_keys, raw = generate_samples(base_seed, node_id, item_key,
                                       sample_count, group_count)
    values = raw * float(scale_value)
    return aggregate(group_keys, values, aggregations)


def deterministic_elapsed_ms(node_id: str, item_key: str, version: int,
                             request_id: str, work_min: int,
                             work_max: int) -> int:
    """Deterministic integer elapsed time in milliseconds.

    The node still performs a real (short) busy loop per request, but the
    reported elapsed time is a stable function of the request identity so that
    regression comparisons are reproducible.  Values are bounded by the
    registry-declared ``work_ms_*`` window.
    """

    if work_max <= work_min:
        return int(work_min)
    span = work_max - work_min + 1
    return int(work_min + stable_hash(node_id, item_key, version,
                                      request_id) % span)

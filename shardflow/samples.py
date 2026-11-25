"""Deterministic integer sample generation and fixed-point aggregation.

Samples are produced entirely locally from the integer seed declared on the
node. Every arithmetic step uses NumPy int64 arrays; no floating point value
ever enters the aggregate, which makes outcomes byte-comparable across runs and
machines.
"""

import numpy as np

from .util import canonical_json, sha256_hex

GROUP_COUNT = 8
GROUP_MOD = 1000
VALUE_MOD = 10_000


def generate_samples(seed, sample_count):
    """Return ``(groups, values)`` int64 arrays of ``sample_count`` rows."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, GROUP_COUNT, size=sample_count, dtype=np.int64)
    values = rng.integers(0, VALUE_MOD, size=sample_count, dtype=np.int64)
    return groups, values


def sample_digest(groups, values):
    """Stable digest over the generated sample batch."""
    return sha256_hex(
        {
            "groups": [int(v) for v in groups],
            "values": [int(v) for v in values],
        }
    )


def fixed_point_aggregate(groups, values, scale, config_values, compute_iters):
    """Group values and compute integer fixed-point weighted aggregates.

    ``config_values`` is the effective integer mapping for the configuration
    revision. All arithmetic stays in int64. The deterministic burn-in loop
    gives long-running batches a real, measurable compute duration.
    """
    groups = np.asarray(groups, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    weight = np.int64(scale)
    accumulator = np.zeros(GROUP_COUNT, dtype=np.int64)
    counts = np.zeros(GROUP_COUNT, dtype=np.int64)
    # np.add.at is unbuffered and therefore deterministic for repeated indices.
    np.add.at(accumulator, groups, values)
    np.add.at(counts, groups, np.ones_like(values))
    accumulator *= weight

    revision_factor = np.int64(1)
    for key in sorted(config_values):
        revision_factor = revision_factor * np.int64(config_values[key])
    # Apply a deterministic revision-dependent factor without overflowing for
    # the bounded demo values.
    accumulator *= (revision_factor % np.int64(7919))

    total = np.int64(0)
    for _ in range(max(0, int(compute_iters))):
        total += accumulator.sum()
    average = np.zeros(GROUP_COUNT, dtype=np.int64)
    nonzero = counts > 0
    average[nonzero] = accumulator[nonzero] // counts[nonzero]

    rows = []
    for group_id in range(GROUP_COUNT):
        rows.append(
            {
                "group": group_id,
                "count": int(counts[group_id]),
                "sum": int(accumulator[group_id]),
                "average": int(average[group_id]),
            }
        )
    summary = {
        "scale": int(scale),
        "sample_count": int(values.size),
        "group_count": GROUP_COUNT,
        "rows": rows,
        "burn_total": int(total),
        "revision_factor_mod": int(revision_factor % np.int64(7919)),
    }
    summary["digest"] = sha256_hex(
        {key: summary[key] for key in sorted(summary)}
    )
    return summary


def encode_batch(groups, values):
    """JSON-safe integer lists for the JSON-lines wire protocol."""
    return {
        "groups": [int(v) for v in groups],
        "values": [int(v) for v in values],
    }


def decode_batch(batch):
    groups = np.asarray(batch["groups"], dtype=np.int64)
    values = np.asarray(batch["values"], dtype=np.int64)
    if groups.shape != values.shape or groups.ndim != 1:
        raise ValueError("sample arrays must be one dimensional and equal length")
    return groups, values


def summary_canonical(summary):
    """Canonical serialization used for byte-level outcome comparison."""
    return canonical_json(summary)

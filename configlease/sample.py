"""Deterministic numeric samples and NumPy aggregation.

The sample attached to a revision is derived purely from its integer
``rev_seq`` through a fixed NumPy ``default_rng`` seed.  Therefore any node,
any interleaving of applications and any replay produces byte for byte the
same aggregate.  Only integer arithmetic and fixed size floats are emitted so
the summary survives JSON round trips and cross process comparison.
"""
from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE_SIZE = 64
RNG_SEED_BASE = 20261121


def sample_for_revision(rev_seq: int) -> np.ndarray:
    """Return the deterministic float sample owned by a revision number."""
    seed = RNG_SEED_BASE + int(rev_seq) * 7919
    rng = np.random.default_rng(seed)
    return rng.normal(loc=float(rev_seq), scale=1.5, size=SAMPLE_SIZE)


def summarize(sample: np.ndarray) -> dict[str, Any]:
    """Aggregate a sample into JSON friendly, deterministic integers/floats."""
    arr = np.asarray(sample, dtype=np.float64)
    rounded = np.round(arr, 9)
    return {
        "count": int(arr.size),
        "sum": int(round(float(np.sum(rounded)) * 1_000_000)),
        "mean": round(float(np.mean(rounded)), 9),
        "min": round(float(np.min(rounded)), 9),
        "max": round(float(np.max(rounded)), 9),
        "std": round(float(np.std(rounded, ddof=0)), 9),
    }


def summary_for_revision(rev_seq: int) -> dict[str, Any]:
    return summarize(sample_for_revision(rev_seq))
